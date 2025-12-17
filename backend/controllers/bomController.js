const { spawn } = require('child_process');
const { execSync } = require('child_process');
const path = require('path');

// Path to your Python script
const PYTHON_SCRIPT = path.join(__dirname, '../python/bom_api.py');


/**
 * Get Python executable path with fallback to python3
 * Tries: PYTHON_PATH env var -> python3 -> python
 */
const getPythonExecutable = () => {
  // Use explicit PYTHON_PATH if set
  if (process.env.PYTHON_PATH) {
    return process.env.PYTHON_PATH;
  }
  
  // Try to find python3 first (common on macOS/Linux)
  try {
    execSync('which python3', { stdio: 'ignore' });
    return 'python3';
  } catch (e) {
    // python3 not found, try python
    try {
      execSync('which python', { stdio: 'ignore' });
      return 'python';
    } catch (e2) {
      // Neither found, default to python3 (most common)
      return 'python3';
    }
  }
};

/**
 * Execute Python script and return parsed JSON result
 * Python executable is detected at runtime to ensure it's always current
 */
const executePythonScript = (scriptPath, args = []) => {
  return new Promise((resolve, reject) => {
    // Get Python executable at runtime (not at module load) to avoid caching issues
    const PYTHON_EXECUTABLE = getPythonExecutable();
    
    const python = spawn(PYTHON_EXECUTABLE, [scriptPath, ...args]);
    
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
        reject(new Error(`Failed to parse Python output: ${error.message}`));
      }
    });

    python.on('error', (error) => {
      reject(new Error(`Failed to start Python process: ${error.message}. Tried executable: ${PYTHON_EXECUTABLE}. Please ensure Python 3 is installed.`));
    });
  });
};

/**
 * Generate BOM from BOQ payload
 */
exports.generateBOM = async (req, res) => {
  try {
    const payload = req.body;

    // Validate required fields
    if (!payload.boq || !payload.boq.category) {
      return res.status(400).json({
        success: false,
        message: 'Invalid payload: boq and category are required'
      });
    }

    // Pass payload as JSON string argument
    const result = await executePythonScript(PYTHON_SCRIPT, [
      'generate',
      JSON.stringify(payload)
    ]);

    res.status(200).json({
      success: true,
      data: result
    });

  } catch (error) {
    console.error('Error generating BOM:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to generate BOM',
      error: error.message
    });
  }
};

/**
 * Upload CSV data to DynamoDB
 */
exports.uploadCSVData = async (req, res) => {
  try {
    const { csvFolder } = req.body;

    if (!csvFolder) {
      return res.status(400).json({
        success: false,
        message: 'CSV folder path is required'
      });
    }

    const result = await executePythonScript(PYTHON_SCRIPT, [
      'upload',
      csvFolder
    ]);

    res.status(200).json({
      success: true,
      message: 'CSV data uploaded successfully',
      data: result
    });

  } catch (error) {
    console.error('Error uploading CSV data:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to upload CSV data',
      error: error.message
    });
  }
};

/**
 * Get BOM by BOQ ID (stub - implement storage if needed)
 */
exports.getBOMByBoqId = async (req, res) => {
  try {
    const { boqId } = req.params;

    // This is a placeholder - implement DynamoDB query if you want to store BOMs
    res.status(200).json({
      success: true,
      message: 'Feature not implemented yet',
      boqId
    });

  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Failed to retrieve BOM',
      error: error.message
    });
  }
};

/**
 * Generate BOM for multiple BOQs
 */
exports.generateBatchBOM = async (req, res) => {
  try {
    const { boqs } = req.body;

    if (!Array.isArray(boqs) || boqs.length === 0) {
      return res.status(400).json({
        success: false,
        message: 'boqs array is required and must not be empty'
      });
    }

    const result = await executePythonScript(PYTHON_SCRIPT, [
      'batch',
      JSON.stringify(boqs)
    ]);

    res.status(200).json({
      success: true,
      data: result
    });

  } catch (error) {
    console.error('Error generating batch BOM:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to generate batch BOM',
      error: error.message
    });
  }
};

/**
 * Get material substitutions
 */
exports.getSubstitutions = async (req, res) => {
  try {
    const { materialName } = req.params;

    const result = await executePythonScript(PYTHON_SCRIPT, [
      'substitutions',
      materialName
    ]);

    res.status(200).json({
      success: true,
      data: result
    });

  } catch (error) {
    console.error('Error getting substitutions:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to get substitutions',
      error: error.message
    });
  }
};

/**
 * Predict wastage for a material
 */
exports.predictWastage = async (req, res) => {
  try {
    const materialData = req.body;

    const result = await executePythonScript(PYTHON_SCRIPT, [
      'wastage',
      JSON.stringify(materialData)
    ]);

    res.status(200).json({
      success: true,
      data: result
    });

  } catch (error) {
    console.error('Error predicting wastage:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to predict wastage',
      error: error.message
    });
  }
};