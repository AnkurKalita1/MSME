const { spawn } = require('child_process');
const { execSync } = require('child_process');
const path = require('path');

// Path to Python BOQ script
const BOQ_SCRIPT = path.join(__dirname, '../python/boq_rate_api.py');

/**
 * Get Python executable path with fallback to python3/python
 * Supports both Windows and Unix systems
 * Tries: PYTHON_PATH env var -> python3 -> python -> python3 (default)
 */
const getPythonExecutable = () => {
  // Use explicit PYTHON_PATH if set
  if (process.env.PYTHON_PATH) {
    return process.env.PYTHON_PATH;
  }
  
  const isWindows = process.platform === 'win32';
  const findCommand = isWindows ? 'where' : 'which';
  
  // Try to find python3 first (common on macOS/Linux, also available on Windows)
  try {
    execSync(`${findCommand} python3`, { stdio: 'ignore' });
    return 'python3';
  } catch (e) {
    // python3 not found, try python
    try {
      execSync(`${findCommand} python`, { stdio: 'ignore' });
      return 'python';
    } catch (e2) {
      // On Windows, also try 'py' launcher
      if (isWindows) {
        try {
          execSync(`${findCommand} py`, { stdio: 'ignore' });
          return 'py';
        } catch (e3) {
          // Fall through to default
        }
      }
      // Neither found, default based on platform
      return isWindows ? 'python' : 'python3';
    }
  }
};

/**
 * Execute Python BOQ script
 * Python executable is detected at runtime to ensure it's always current
 */
const executeBOQScript = (scriptPath, args = []) => {
  return new Promise((resolve, reject) => {
    // Get Python executable at runtime (not at module load) to avoid caching issues
    const PYTHON_EXECUTABLE = getPythonExecutable();
    
    const python = spawn(PYTHON_EXECUTABLE, [scriptPath, ...args], {
      env: {
        ...process.env
      }
    });
    
    let dataString = '';
    let errorString = '';

    python.stdout.on('data', (data) => {
      dataString += data.toString();
    });

    python.stderr.on('data', (data) => {
      errorString += data.toString();
    });

    python.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python process exited with code ${code}: ${errorString}`));
        return;
      }

      try {
        const result = JSON.parse(dataString);
        resolve(result);
      } catch (error) {
        reject(new Error(`Failed to parse Python output: ${error.message}\nOutput: ${dataString}`));
      }
    });

    python.on('error', (error) => {
      reject(new Error(`Failed to start Python process: ${error.message}. Tried executable: ${PYTHON_EXECUTABLE}. Please ensure Python 3 is installed.`));
    });
  });
};

/**
 * Train the rate prediction model
 */
exports.trainModel = async (req, res) => {
  try {
    const result = await executeBOQScript(BOQ_SCRIPT, ['train']);

    res.status(200).json({
      success: true,
      message: 'Model trained successfully',
      data: result
    });

  } catch (error) {
    console.error('Error training model:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to train model',
      error: error.message
    });
  }
};

/**
 * Predict rates for BOQ items
 */
exports.predictRates = async (req, res) => {
  try {
    const payload = req.body;

    // Validate
    if (!payload.items || !Array.isArray(payload.items) || payload.items.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'Invalid payload: items array is required and must not be empty'
      });
    }

    // Validate each item has material
    for (const item of payload.items) {
      if (!item.material) {
        return res.status(400).json({
          success: false,
          message: 'Each item must have a material field'
        });
      }
    }

    const result = await executeBOQScript(BOQ_SCRIPT, [
      'predict',
      JSON.stringify(payload)
    ]);

    res.status(200).json({
      success: true,
      data: result
    });

  } catch (error) {
    console.error('Error predicting rates:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to predict rates',
      error: error.message
    });
  }
};

/**
 * Extract BOQ from text and predict rates
 */
exports.extractAndPredict = async (req, res) => {
  try {
    const { text } = req.body;

    if (!text) {
      return res.status(400).json({
        success: false,
        message: 'Text is required'
      });
    }

    const result = await executeBOQScript(BOQ_SCRIPT, [
      'extract',
      text
    ]);

    res.status(200).json({
      success: true,
      data: result
    });

  } catch (error) {
    console.error('Error extracting BOQ:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to extract BOQ',
      error: error.message
    });
  }
};

/**
 * Health check for BOQ service
 */
exports.boqHealthCheck = async (req, res) => {
  try {
    res.status(200).json({
      success: true,
      message: 'BOQ service is running',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'BOQ service health check failed',
      error: error.message
    });
  }
};