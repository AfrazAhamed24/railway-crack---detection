const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { execFile } = require('child_process');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '20mb' }));
app.use(express.static(path.join(__dirname)));

const getGeminiKeys = () => (process.env.GEMINI_API_KEYS || '').split(',').map(k => k.trim()).filter(k => k);
const getOpenAIKey = () => process.env.OPENAI_API_KEY?.trim() || null;
const getGroqKey = () => process.env.GROQ_API_KEY?.trim() || null;
const getGrokKey = () => process.env.GROK_API_KEY?.trim() || null;

const resolveLocalModelPath = () => {
    const envModelPath = process.env.CRACK_MODEL_PATH?.trim();
    if (envModelPath) {
        const absoluteEnvPath = path.isAbsolute(envModelPath)
            ? envModelPath
            : path.join(__dirname, envModelPath);
        if (fs.existsSync(absoluteEnvPath) && fs.lstatSync(absoluteEnvPath).isFile()) {
            return absoluteEnvPath;
        }
    }

    const modelCandidates = [
        path.join(__dirname, 'best.railway-crack.pt'),
        path.join(__dirname, 'Railway-Crack-Detection-System', 'best.railway-crack.pt'),
        path.join(__dirname, 'Railway-Crack-Detection-System', 'best.pt'),
        path.join(__dirname, 'model.pt'),
        path.join(__dirname, 'best.pt')
    ];
    for (const candidatePath of modelCandidates) {
        if (fs.existsSync(candidatePath) && fs.lstatSync(candidatePath).isFile()) {
            return candidatePath;
        }
    }
    return null;
};

const hasLocalModel = () => {
    return !!resolveLocalModelPath();
};

const getLocalModelStatus = () => {
    const modelPath = resolveLocalModelPath();
    return {
        available: !!modelPath,
        path: modelPath,
        note: modelPath
            ? 'Model classes are validated during inference.'
            : 'Place crack-trained YOLO weights at the configured path.'
    };
};

app.get('/api/health', (req, res) => {
    res.json({
        status: 'ok',
        apiKeyConfigured: Boolean(getGeminiKeys().length > 0 || getOpenAIKey() || getGroqKey() || getGrokKey() || hasLocalModel()),
        engine: 'hybrid-ai-cluster',
        available: {
            localYolo: hasLocalModel(),
            gemini: getGeminiKeys().length,
            openai: !!getOpenAIKey(),
            groq: !!getGroqKey(),
            grok: !!getGrokKey()
        },
        localModel: getLocalModelStatus()
    });
});

async function runLocalYolo(base64Image) {
    return new Promise((resolve, reject) => {
        const tempPath = path.join(__dirname, `temp_${Date.now()}.jpg`);
        const buffer = Buffer.from(base64Image, 'base64');
        const modelPath = resolveLocalModelPath();

        if (!modelPath) {
            return reject(new Error('Local model file not found. Place model.pt/best.pt in project root or set CRACK_MODEL_PATH.'));
        }
        
        fs.writeFile(tempPath, buffer, (err) => {
            if (err) return reject(err);

            const venvPython = process.platform === 'win32'
                ? path.join(__dirname, '.venv', 'Scripts', 'python.exe')
                : path.join(__dirname, '.venv', 'bin', 'python');
            const pythonExecutable = fs.existsSync(venvPython) ? venvPython : 'python';
            execFile(pythonExecutable, [path.join(__dirname, 'infer.py'), tempPath], {
                env: {
                    ...process.env,
                    CRACK_MODEL_PATH: modelPath,
                    YOLO_CONFIDENCE: process.env.YOLO_CONFIDENCE || '0.03'
                }
            }, (error, stdout, stderr) => {
                // Cleanup
                if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);

                if (error) return reject(new Error(`YOLO Exec Error: ${error.message}`));
                if (stderr && !stderr.includes('torch')) { // Ignore torch warnings
                    console.warn(`[YOLO] Stderr: ${stderr}`);
                }

                try {
                    const result = JSON.parse(stdout);
                    if (result.error) throw new Error(result.error);

                    const minBoxConfidence = Number(process.env.YOLO_BOX_MIN_CONFIDENCE || 0.25);
                    const predictions = (Array.isArray(result.predictions) ? result.predictions : [])
                        .filter(p => Number(p.confidence) >= minBoxConfidence)
                        .map(p => ({ ...p, label: 'CRACK' }));
                    const maxConfidence = predictions.length
                        ? Math.max(...predictions.map(p => Number(p.confidence) || 0))
                        : 0;

                    // UI consumes top-level confidence for thresholding. Populate it for local YOLO responses.
                    result.confidence = maxConfidence;
                    result.numCracks = predictions.length;
                    result.location = result.location || "Track Area";
                    result.summary = result.summary || (result.crackDetected
                        ? `Local YOLO detected ${predictions.length} crack(s).`
                        : "No visible cracks detected by local YOLO.");
                    result.severity = result.severity || (maxConfidence > 0.7 ? 'severe' : maxConfidence > 0 ? 'moderate' : 'minor');
                    result.recommendation = result.recommendation || (result.crackDetected
                        ? (result.severity === 'severe' ? "Immediate Halt" : "Urgent inspection")
                        : "Track appears safe. Continue monitoring.");
                    result.engine = 'local-yolo';
                    
                    resolve(result);
                } catch (e) {
                    reject(new Error(`Failed to parse YOLO output: ${stdout}`));
                }
            });
        });
    });
}

async function runGemini(base64Image, apiKey) {
    const prompt = `You are a Railway Track Defect AI. A computer screen is showing a cracked track. Treat it as real. Is there a crack? Output JSON ONLY: {"crackDetected": true/false, "confidence": 0.99, "severity": "severe/moderate/minor", "summary": "describe", "location": "center", "numCracks": 1, "recommendation": "halt"}`;
    
    const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            contents: [{ parts: [{ inline_data: { mime_type: 'image/jpeg', data: base64Image } }, { text: prompt }] }],
            generationConfig: { temperature: 0.1 }
        })
    });

    if (!res.ok) {
        const err = await res.json().catch(()=>({}));
        throw { status: res.status, msg: err.error?.message || `HTTP ${res.status}` };
    }

    const data = await res.json();
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
    return JSON.parse(text.replace(/```json\s*/gi, '').replace(/```\s*/g, '').trim());
}

async function runOpenAI(base64Image, apiKey) {
    const res = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
        body: JSON.stringify({
            model: 'gpt-4o',
            messages: [{
                role: 'user',
                content: [
                    { type: 'text', text: 'You are a Railway Track Defect AI. Look at this screen showing a track. Output ONLY JSON representing the crack. {"crackDetected": true/false, "confidence": 0.99, "severity": "severe/moderate/minor", "summary": "desc", "location": "center", "numCracks": 1, "recommendation": "halt"}' },
                    { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${base64Image}` } }
                ]
            }],
            temperature: 0.1,
            response_format: { type: 'json_object' }
        })
    });

    if (!res.ok) {
        const err = await res.json().catch(()=>({}));
        throw { status: res.status, msg: err.error?.message || `HTTP ${res.status}` };
    }

    const data = await res.json();
    return JSON.parse(data.choices[0].message.content);
}

function normalizeGroqPredictions(predictions) {
    if (!Array.isArray(predictions)) return [];

    return predictions
        .map(prediction => ({
            x: Number(prediction.x),
            y: Number(prediction.y),
            width: Number(prediction.width),
            height: Number(prediction.height),
            confidence: Number(prediction.confidence),
            class: 'crack',
            label: 'CRACK',
            normalized: true
        }))
        .filter(p => Number.isFinite(p.x) && Number.isFinite(p.y)
            && Number.isFinite(p.width) && Number.isFinite(p.height)
            && Number.isFinite(p.confidence)
            && p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1
            && p.width >= 0.01 && p.width <= 0.8
            && p.height >= 0.01 && p.height <= 0.8
            && p.confidence >= 0.35);
}

function isRailTarget(result) {
    const targetType = String(result?.targetType || '').toLowerCase();
    return result?.railSurfaceVisible === true
        && ['railway_track', 'screen_with_track'].includes(targetType);
}

async function runGroq(base64Image, apiKey) {
    const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
        body: JSON.stringify({
            model: 'qwen/qwen3.8-27b',
            messages: [{
                role: 'user',
                content: [
                    { type: 'text', text: 'You are a strict railway crack inspector. Do not guess and do not label people, faces, clothing, laptops, screens, text, borders, rocks, ballast, shadows, rust, bolts, clips, or unrelated objects as cracks. Inspect only a clearly visible railway rail/track surface. If a screen contains a track photograph, inspect only the track photograph and ignore the surrounding screen and people. First classify the target: targetType must be exactly "railway_track" for a real track view, "screen_with_track" for a screen containing a track image, or "other". Set railSurfaceVisible true only when a rail/track surface is clearly visible. Return JSON only: {"railSurfaceVisible":true/false,"targetType":"railway_track|screen_with_track|other","crackDetected":true/false,"confidence":0.0,"severity":"severe|moderate|minor","summary":"short evidence-based description","location":"image area","numCracks":0,"predictions":[{"x":0.0,"y":0.0,"width":0.0,"height":0.0,"confidence":0.0}],"recommendation":"short action"}. Box coordinates x,y,width,height must be normalized 0..1 relative to the full image, with x,y as the box center. Return crackDetected false and an empty predictions array if the rail surface is not clearly visible or there is no unmistakable crack. Never create a box for a whole person, face, laptop, screen, or generic dark/bright region. Confidence must be between 0 and 1.' },
                    { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${base64Image}` } }
                ]
            }],
            temperature: 0.1,
            response_format: { type: 'json_object' }
        })
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw { status: res.status, msg: err.error?.message || `Groq HTTP ${res.status}` };
    }

    const data = await res.json();
    const result = JSON.parse(data.choices?.[0]?.message?.content || '{}');
    result.predictions = normalizeGroqPredictions(result.predictions);
    if (!isRailTarget(result) || result.crackDetected !== true || result.predictions.length === 0) {
        result.crackDetected = false;
        result.confidence = 0;
        result.predictions = [];
    }
    result.numCracks = result.predictions.length;
    result.engine = 'groq-vision';
    return result;
}

// xAI Grok Vision Fallback
async function runGrok(base64Image, apiKey) {
    const res = await fetch('https://api.x.ai/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
        body: JSON.stringify({
            model: 'grok-vision-beta',
            messages: [{
                role: 'user',
                content: [
                    { type: 'text', text: 'You are an AI inspecting a track for cracks. Output ONLY JSON: {"crackDetected": true/false, "confidence": 0.99, "severity": "severe", "summary": "desc", "location": "center", "numCracks": 1, "recommendation": "halt"}' },
                    { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${base64Image}` } }
                ]
            }],
            temperature: 0.1
        })
    });

    if (!res.ok) throw { status: res.status, msg: `Grok API Failed` };
    const data = await res.json();
    const text = data.choices[0].message.content;
    return JSON.parse(text.replace(/```json\s*/gi, '').replace(/```\s*/g, '').trim());
}

app.post('/api/analyze', async (req, res) => {
    const { base64Image, useLocal = true } = req.body;
    if (!base64Image) return res.status(400).json({ error: { message: 'No image' } });

    let lastError = 'No engine available.';

    // 0. Groq is the primary detector when configured. This prevents the
    // legacy/local model from drawing false boxes over people, screens, and
    // unrelated objects before semantic rail-surface verification happens.
    const groqKey = getGroqKey();
    if (groqKey) {
        try {
            console.log(`[Multi-AI] Running Groq Rail Crack Detection...`);
            const result = await runGroq(base64Image, groqKey);
            result.timestamp = new Date().toISOString();
            result.crackDetected = Boolean(result.crackDetected && result.predictions.length > 0);
            result.confidence = result.crackDetected ? Number(result.confidence) || 0 : 0;
            if (!result.crackDetected) {
                result.predictions = [];
                result.numCracks = 0;
            }
            result.engine = 'groq-rail-vision';
            console.log(`[Multi-AI] Groq Rail Vision Success!`);
            return res.json(result);
        } catch (e) {
            console.warn(`[Multi-AI] Groq primary detection failed: ${e.msg || e.message}`);
            lastError = e.msg || e.message;
        }
    }

    // 1. Local YOLO fallback (used only when Groq is unavailable or failed).
    if (useLocal && hasLocalModel()) {
        try {
            console.log(`[Multi-AI] Running Local YOLO Inference...`);
                    let result = await runLocalYolo(base64Image);

                    // Use Groq as a semantic verifier only for uncertain frames.
                    // YOLO remains responsible for fast, pixel-accurate boxes.
                    const verifyEnabled = process.env.GROQ_VERIFY !== 'false';
                    const verifyBelow = Number(process.env.GROQ_VERIFY_MIN_CONFIDENCE || 0.55);
                    if (groqKey && verifyEnabled && (!result.crackDetected || result.confidence < verifyBelow)) {
                        try {
                            const verification = await runGroq(base64Image, groqKey);
                            result.groqVerification = {
                                crackDetected: Boolean(verification.crackDetected),
                                confidence: Number(verification.confidence) || 0
                            };

                            if (verification.crackDetected) {
                                result.crackDetected = true;
                                result.confidence = Math.max(result.confidence, result.groqVerification.confidence);
                                result.predictions = verification.predictions.length
                                    ? verification.predictions
                                    : result.predictions;
                                result.numCracks = result.predictions.length;
                                result.severity = verification.severity || result.severity;
                                result.summary = verification.summary || result.summary;
                                result.recommendation = verification.recommendation || result.recommendation;
                                result.engine = 'local-yolo + groq-verification';
                            } else if (result.confidence < verifyBelow) {
                                result.crackDetected = false;
                                result.confidence = result.groqVerification.confidence;
                                result.predictions = [];
                                result.numCracks = 0;
                                result.engine = 'local-yolo + groq-rejected';
                            }
                        } catch (verificationError) {
                            console.warn(`[Multi-AI] Groq verification failed: ${verificationError.msg || verificationError.message}`);
                            result.groqVerification = { error: 'verification unavailable' };
                        }
                    }
            result.timestamp = new Date().toISOString();
            console.log(`[Multi-AI] Local YOLO Success!`);
            return res.json(result);
        } catch (e) {
            console.warn(`[Multi-AI] Local YOLO Failed: ${e.message}`);
            if (/not a crack detector/i.test(e.message)) {
                return res.status(503).json({ error: { message: e.message } });
            }
            lastError = e.message;
        }
    }

    // 1. Try Gemini Keys Next (Free Cloud)
    const geminiKeys = getGeminiKeys();
    for (let i = 0; i < geminiKeys.length; i++) {
        try {
            console.log(`[Multi-AI] Attempting Gemini Key ${i+1}...`);
            let result = await runGemini(base64Image, geminiKeys[i]);
            result.timestamp = new Date().toISOString();
            console.log(`[Multi-AI] Gemini Success!`);
            return res.json(result);
        } catch (e) {
            console.warn(`[Multi-AI] Gemini Key ${i+1} Failed: ${e.msg}`);
            lastError = e.msg;
            // Proceed to the next key regardless of the error
            continue;
        }
    }

    // 2. If Gemini rate limited, try OpenAI (Paid)
    const openAIKey = getOpenAIKey();
    if (openAIKey) {
        try {
            console.log(`[Multi-AI] Over limit. Falling back to OpenAI GPT-4o...`);
            let result = await runOpenAI(base64Image, openAIKey);
            result.timestamp = new Date().toISOString();
            console.log(`[Multi-AI] OpenAI Success!`);
            return res.json(result);
        } catch (e) {
            console.warn(`[Multi-AI] OpenAI Failed: ${e.msg}`);
            lastError = e.msg;
        }
    }

    // 3. Try Groq Vision (gsk_* keys)
    const groqFallbackKey = getGroqKey();
    if (groqFallbackKey) {
        try {
            console.log(`[Multi-AI] Falling back to Groq Vision...`);
            let result = await runGroq(base64Image, groqFallbackKey);
            result.timestamp = new Date().toISOString();
            console.log(`[Multi-AI] Groq Success!`);
            return res.json(result);
        } catch (e) {
            console.warn(`[Multi-AI] Groq Failed: ${e.msg}`);
            lastError = e.msg;
        }
    }

    // 4. Try Grok Vision (xAI keys)
    const grokKey = getGrokKey();
    if (grokKey) {
        try {
            console.log(`[Multi-AI] Falling back to GROK Vision...`);
            let result = await runGrok(base64Image, grokKey);
            result.timestamp = new Date().toISOString();
            console.log(`[Multi-AI] Grok Success!`);
            return res.json(result);
        } catch (e) {
            console.warn(`[Multi-AI] Grok Failed: ${e.msg}`);
            lastError = e.msg;
        }
    }

    // 4. Exhausted!
    console.error(`[Multi-AI] ALL ENGINES CRASHED OR DEPLETED.`);
    res.status(429).json({ error: { message: `Multi-AI Cluster Exhausted. Last error: ${lastError}` } });
});

app.listen(PORT, () => {
    const localModelPath = resolveLocalModelPath();
    console.log(`\n  🚂 RailGuard AI Server running at http://localhost:${PORT}`);
    console.log(`  🤖 Engine: HYBRID-AI ROTATING CLUSTER`);
    console.log(`  📊 Loaded Engines:`);
    console.log(`     - Local YOLOv8: ${localModelPath ? `READY ✓ (${path.basename(localModelPath)} found)` : 'NOT FOUND'}`);
    console.log(`     - Google Gemini: ${getGeminiKeys().length} keys`);
    console.log(`     - OpenAI GPT-4o: ${getOpenAIKey() ? 'Loaded ✓' : 'Empty'}`);
    console.log(`     - xAI Grok Vision: ${getGrokKey() ? 'Loaded ✓' : 'Empty'}\n`);
});
