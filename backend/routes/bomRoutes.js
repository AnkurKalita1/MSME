const express = require('express');
const router = express.Router();
const bomController = require('../controllers/bomController');

// Generate BOM from BOQ payload
router.post('/generate', bomController.generateBOM);



// Upload CSV data to DynamoDB
router.post('/upload-data', bomController.uploadCSVData);

// Get BOM by BOQ ID (if you want to store and retrieve)
router.get('/:boqId', bomController.getBOMByBoqId);

// Batch BOM generation
router.post('/generate-batch', bomController.generateBatchBOM);

// Get material substitutions
router.get('/substitutions/:materialName', bomController.getSubstitutions);

// Get wastage prediction for a material
router.post('/predict-wastage', bomController.predictWastage);

// Get material suggestions
router.get('/material-suggestions', bomController.getMaterialSuggestions);

module.exports = router;