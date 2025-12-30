const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
require('dotenv').config();

const bomRoutes = require('./routes/bomRoutes');
const wbsRoutes = require('./routes/wbsRoutes');
const boqRoutes = require('./routes/boqRoutes');
const boqCleaningRoutes = require('./routes/boqCleaningRoutes');
const predictiveRoutes = require('./routes/predictiveRoutes');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(helmet());
app.use(cors());
app.use(morgan('dev'));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'OK',
    timestamp: new Date().toISOString(),
    service: 'BOM Generation API'
  });
});

// API Routes
app.use('/api/bom', bomRoutes);
app.use('/api/wbs', wbsRoutes);
app.use('/api/boq', boqRoutes);
app.use('/api/boq-cleaning', boqCleaningRoutes);
app.use('/api/predictive', predictiveRoutes);



// 404 Handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: 'Route not found'
  });
});

// Global Error Handler
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(err.status || 500).json({
    success: false,
    message: err.message || 'Internal Server Error',
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 BOM API Server running on port ${PORT}`);
  console.log(`📊 Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`🔗 Health check: http://localhost:${PORT}/health`);
  console.log(`🐍 Python detection will be logged when controllers are loaded`);
});

module.exports = app;