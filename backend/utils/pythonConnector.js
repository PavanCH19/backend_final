const { spawn } = require('child_process');
const path = require('path');
// ============================================
// PYTHON EXECUTOR (ENHANCED)
// ============================================

function executePythonModel(modelConfig, functionName, data = {}, timeout = 30000) {
    return new Promise((resolve, reject) => {
        const bridgePath = path.join(__dirname, '..', 'python_models', 'bridge.py');

        // Spawn Python process with UTF-8 encoding
        const python = spawn(modelConfig.pythonPath, [bridgePath], {
            env: {
                ...process.env,
                ...modelConfig.envVars,
                PYTHONIOENCODING: 'utf-8'  // Force UTF-8 encoding for Windows
            },
            cwd: __dirname
        });

        let output = '';
        let errorOutput = '';
        let timeoutId;

        // Set timeout
        if (timeout > 0) {
            timeoutId = setTimeout(() => {
                python.kill();
                reject({
                    success: false,
                    error: 'Python execution timeout',
                    timeout: timeout
                });
            }, timeout);
        }

        // Auto-fix relative paths (ensure no 'utils' prefix issue)
        let scriptPath = modelConfig.scriptPath;
        if (!path.isAbsolute(scriptPath)) {
            scriptPath = path.join(__dirname, '..', scriptPath);
        }

        // Prepare request payload
        const request = {
            script_path: scriptPath,
            function: functionName,
            data
        };

        // Send data to Python
        python.stdin.write(JSON.stringify(request));
        python.stdin.end();

        python.stdout.on('data', (chunk) => {
            output += chunk.toString();
        });

        python.stderr.on('data', (chunk) => {
            errorOutput += chunk.toString();
            console.error('Python stderr:', chunk.toString());
        });

        python.on('close', (code) => {
            clearTimeout(timeoutId);

            if (code !== 0) {
                return reject({
                    success: false,
                    error: 'Python execution failed',
                    exitCode: code,
                    stderr: errorOutput,
                    stdout: output
                });
            }

            try {
                const lines = output.trim().split('\n').filter(Boolean);
                const jsonLine = lines[lines.length - 1];
                const result = JSON.parse(jsonLine);

                if (result.success) {
                    resolve(result);
                } else {
                    reject(result);
                }
            } catch (e) {
                reject({
                    success: false,
                    error: 'Invalid JSON from Python',
                    parseError: e.message,
                    rawOutput: output
                });
            }
        });

        python.on('error', (err) => {
            clearTimeout(timeoutId);
            reject({
                success: false,
                error: 'Failed to start Python process',
                details: err.message
            });
        });
    });
}

module.exports = { executePythonModel };