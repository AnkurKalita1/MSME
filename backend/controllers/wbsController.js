const { spawn } = require('child_process');
const { execSync } = require('child_process');
const path = require('path');

// Path to Python WBS script
const WBS_SCRIPT = path.join(__dirname, '../python/wbs_api.py');

/**
 * Get Python executable path with fallback to python3/python
 * Supports both Windows and Unix systems
 * Tries: PYTHON_PATH env var -> python3 -> python -> python3 (default)
 * This is called at runtime to avoid module caching issues
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
 * Execute Python WBS script
 * Python executable is detected at runtime to ensure it's always current
 */
const executeWBSScript = (scriptPath, args = []) => {
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
 * Generate WBS from BOQ payload
 */
exports.generateWBS = async (req, res) => {
  try {
    const payload = req.body;

    // Validate required fields
    if (!payload.boq_text) {
      return res.status(400).json({
        success: false,
        message: 'Invalid payload: boq_text is required'
      });
    }

    // Set defaults for optional fields
    const wbsPayload = {
      boq_text: payload.boq_text,
      quantity: payload.quantity || 1.0,
      region: payload.region || 'central',
      season: payload.season || 'normal',
      grade: payload.grade || 'B',
      rate_most: payload.rate_most || 0.0,
      rate_min: payload.rate_min || 0.0,
      rate_max: payload.rate_max || 0.0,
      confidence: payload.confidence || 1.0,
      subcategory: payload.subcategory || null
    };

    // Pass payload as JSON string argument
    const result = await executeWBSScript(WBS_SCRIPT, [
      'generate',
      JSON.stringify(wbsPayload)
    ]);

    res.status(200).json({
      success: true,
      data: result
    });

  } catch (error) {
    console.error('Error generating WBS:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to generate WBS',
      error: error.message
    });
  }
};

/**
 * Generate WBS for multiple BOQ items (batch)
 */
exports.generateBatchWBS = async (req, res) => {
  try {
    const { boqs } = req.body;

    if (!Array.isArray(boqs) || boqs.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'boqs array is required and must not be empty'
      });
    }

    const results = [];

    for (const boq of boqs) {
      try {
        const wbsPayload = {
          boq_text: boq.boq_text,
          quantity: boq.quantity || 1.0,
          region: boq.region || 'central',
          season: boq.season || 'normal',
          grade: boq.grade || 'B',
          rate_most: boq.rate_most || 0.0,
          rate_min: boq.rate_min || 0.0,
          rate_max: boq.rate_max || 0.0,
          confidence: boq.confidence || 1.0,
          subcategory: boq.subcategory || null
        };

        const result = await executeWBSScript(WBS_SCRIPT, [
          'generate',
          JSON.stringify(wbsPayload)
        ]);

        results.push({
          success: true,
          data: result
        });
      } catch (error) {
        results.push({
          success: false,
          error: error.message,
          boq_text: boq.boq_text
        });
      }
    }

    res.status(200).json({
      success: true,
      results: results
    });

  } catch (error) {
    console.error('Error generating batch WBS:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to generate batch WBS',
      error: error.message
    });
  }
};

/**
 * Health check for WBS service
 */
exports.wbsHealthCheck = async (req, res) => {
  try {
    res.status(200).json({
      success: true,
      message: 'WBS service is running',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'WBS service health check failed',
      error: error.message
    });
  }
};