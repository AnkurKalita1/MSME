import { useState, useEffect } from 'react';
import { Home, FileText, ShoppingCart, Activity, CheckCircle, DollarSign, AlertTriangle, TrendingUp, Plus, Calendar, FileBarChart, Loader2, X, Brain, Clock, TrendingDown } from 'lucide-react';
import StatCard from './StatCard';
import QuickActionCard from './QuickActionCard';
import { predictiveAPI } from '../services/api';

const Dashboard = ({ onNavigate }) => {
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState(null);
  const [allPredictions, setAllPredictions] = useState(null);
  const projects = [
    { id: 1, name: 'Office Interior - TechCorp', status: 'on-track', progress: 65, delay: 0, budget: 95 },
    { id: 2, name: 'Residential Villa - Sharma', status: 'at-risk', progress: 40, delay: 5, budget: 108 },
    { id: 3, name: 'Tank Cleaning - Factory A', status: 'delayed', progress: 75, delay: 12, budget: 102 },
    { id: 4, name: 'Retail Fitout - Fashion Store', status: 'on-track', progress: 30, delay: 0, budget: 98 }
  ];

  const statusColors = {
    'on-track': 'bg-green-500',
    'at-risk': 'bg-yellow-500',
    'delayed': 'bg-red-500'
  };

  // Calculate features from BOQ/WBS/BOM data stored in localStorage
  const calculateFeaturesFromData = () => {
    try {
      // Get BOQ items
      const boqItems = JSON.parse(localStorage.getItem('boqItems') || '[]');
      
      // Calculate planned quantities and costs from BOQ
      const planned_qty = boqItems.reduce((sum, item) => sum + (item.qty || 0), 0);
      const planned_cost = boqItems.reduce((sum, item) => sum + ((item.qty || 0) * (item.rate_most_likely || 0)), 0);
      
      // For demo purposes, simulate actual values (in real app, these would come from actual project data)
      // Using 85% of planned as actual to show some variance
      const actual_qty = planned_qty * 0.85;
      const actual_cost = planned_cost * 1.10; // 10% cost overrun
      
      // Calculate variances
      const qty_variance = planned_qty > 0 ? (actual_qty - planned_qty) / planned_qty : 0;
      const cost_variance = planned_cost > 0 ? (actual_cost - planned_cost) / planned_cost : 0;
      
      // Get BOM data if available
      const bomData = JSON.parse(localStorage.getItem('bomData') || '[]');
      const mean_bom_cost = bomData.length > 0 
        ? bomData.reduce((sum, bom) => sum + (bom.totalCost || 0), 0) / bomData.length
        : planned_cost;
      
      return {
        planned_qty: planned_qty || 100,
        actual_qty: actual_qty || 85,
        planned_cost: planned_cost || 50000,
        actual_cost: actual_cost || 55000,
        dpr_total_qty: actual_qty || 85,
        dpr_reports: boqItems.length || 5,
        mean_bom_cost: mean_bom_cost || 52000,
        qty_variance: qty_variance || -0.15,
        cost_variance: cost_variance || 0.10
      };
    } catch (e) {
      console.error('Error calculating features:', e);
      // Return default features if calculation fails
      return {
        planned_qty: 100,
        actual_qty: 85,
        planned_cost: 50000,
        actual_cost: 55000,
        dpr_total_qty: 85,
        dpr_reports: 5,
        mean_bom_cost: 52000,
        qty_variance: -0.15,
        cost_variance: 0.10
      };
    }
  };

  // Automatically fetch all predictions on mount and when data changes
  useEffect(() => {
    const fetchAllPredictions = async () => {
      setPredictionLoading(true);
      setPredictionError(null);

      try {
        const features = calculateFeaturesFromData();
        const response = await predictiveAPI.predictAll(features);
        
        if (response.success && response.data && response.data.predictions) {
          setAllPredictions(response.data.predictions);
        } else {
          setPredictionError(response.error || 'Failed to get predictions');
        }
      } catch (err) {
        setPredictionError(err.message || 'Failed to make predictions');
      } finally {
        setPredictionLoading(false);
      }
    };

    fetchAllPredictions();
    
    // Also fetch when localStorage changes (BOQ/WBS/BOM data updates)
    const handleStorageChange = () => {
      fetchAllPredictions();
    };
    
    window.addEventListener('storage', handleStorageChange);
    
    // Poll for localStorage changes (since storage event only fires in other tabs)
    const interval = setInterval(() => {
      fetchAllPredictions();
    }, 5000); // Refresh every 5 seconds
    
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="grid grid-cols-4 gap-4">
        <StatCard
          title="Active Projects"
          value="12"
          change="+2 this month"
          icon={FileText}
          color="blue"
        />
        <StatCard
          title="On-Time Rate"
          value="75%"
          change="+15% vs last month"
          icon={CheckCircle}
          color="green"
        />
        <StatCard
          title="Budget Health"
          value="₹42.5L"
          change="96% utilization"
          icon={DollarSign}
          color="purple"
        />
        <StatCard
          title="AI Predictions"
          value="3 Risks"
          change="Action needed"
          icon={AlertTriangle}
          color="orange"
        />
      </div>

      {/* AI Predictive Analysis Section */}
      <div className="bg-gradient-to-r from-purple-50 to-indigo-50 border border-purple-200 rounded-lg p-6">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-start gap-3">
            <div className="bg-purple-500 rounded-full p-2 mt-1">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-gray-900 mb-1">AI Predictive Analysis</h3>
              <p className="text-sm text-gray-700">
                Real-time predictions based on your BOQ, WBS, and BOM data
                {predictionLoading && (
                  <span className="ml-2 inline-flex items-center gap-1 text-purple-600">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    Updating...
                  </span>
                )}
              </p>
            </div>
          </div>
        </div>


        {/* Error Message */}
        {predictionError && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3 mb-4">
            <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <p className="text-sm text-red-800">{predictionError}</p>
            </div>
            <button onClick={() => setPredictionError(null)} className="flex-shrink-0">
              <X className="w-4 h-4 text-red-600" />
            </button>
          </div>
        )}

        {/* All Predictions Grid */}
        {allPredictions ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Delay Prediction */}
            {allPredictions.delay && (
              <div className="bg-white border border-purple-200 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <AlertTriangle className="w-5 h-5 text-orange-600" />
                  <h4 className="font-semibold text-gray-900">Delay Prediction</h4>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span className="text-sm text-gray-700">Delay Probability:</span>
                    <span className="font-semibold text-gray-900">
                      {(allPredictions.delay.delay_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span className="text-sm text-gray-700">Will Delay:</span>
                    <span className={`font-semibold ${allPredictions.delay.will_delay ? 'text-red-600' : 'text-green-600'}`}>
                      {allPredictions.delay.will_delay ? 'Yes' : 'No'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span className="text-sm text-gray-700">Risk Level:</span>
                    <span className={`font-semibold ${
                      allPredictions.delay.risk_level === 'High' ? 'text-red-600' :
                      allPredictions.delay.risk_level === 'Medium' ? 'text-yellow-600' :
                      'text-green-600'
                    }`}>
                      {allPredictions.delay.risk_level}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Cost Variance Prediction */}
            {allPredictions.cost && (
              <div className="bg-white border border-purple-200 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <DollarSign className="w-5 h-5 text-blue-600" />
                  <h4 className="font-semibold text-gray-900">Cost Variance</h4>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span className="text-sm text-gray-700">Variance:</span>
                    <span className={`font-semibold ${
                      (allPredictions.cost.cost_variance || 0) > 0 ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {allPredictions.cost.percentage || 
                       ((allPredictions.cost.cost_variance || 0) * 100).toFixed(2) + '%'}
                    </span>
                  </div>
                  {allPredictions.cost.cost_variance && (
                    <div className="text-xs text-gray-500 mt-2">
                      Raw value: {(allPredictions.cost.cost_variance * 100).toFixed(2)}%
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Completion Time Prediction */}
            {allPredictions.completion && (
              <div className="bg-white border border-purple-200 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Clock className="w-5 h-5 text-green-600" />
                  <h4 className="font-semibold text-gray-900">Completion Time</h4>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span className="text-sm text-gray-700">Days to Complete:</span>
                    <span className="font-semibold text-gray-900">
                      {allPredictions.completion.days_to_complete} days
                    </span>
                  </div>
                  {allPredictions.completion.xgb_prediction && (
                    <div className="text-xs text-gray-500 mt-2">
                      <div>XGB: {allPredictions.completion.xgb_prediction} days</div>
                      <div>RF: {allPredictions.completion.rf_prediction} days</div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Profit Margin Prediction */}
            {allPredictions.profit && (
              <div className="bg-white border border-purple-200 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <TrendingUp className="w-5 h-5 text-green-600" />
                  <h4 className="font-semibold text-gray-900">Profit Margin</h4>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span className="text-sm text-gray-700">Margin:</span>
                    <span className={`font-semibold ${
                      (allPredictions.profit.profit_margin || 0) > 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {allPredictions.profit.percentage || 
                       ((allPredictions.profit.profit_margin || 0) * 100).toFixed(2) + '%'}
                    </span>
                  </div>
                  {allPredictions.profit.profit_margin && (
                    <div className="text-xs text-gray-500 mt-2">
                      Raw value: {(allPredictions.profit.profit_margin * 100).toFixed(2)}%
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ) : predictionLoading ? (
          <div className="bg-white border border-purple-200 rounded-lg p-8 text-center">
            <Loader2 className="w-8 h-8 animate-spin text-purple-600 mx-auto mb-2" />
            <p className="text-sm text-gray-600">Loading predictions...</p>
          </div>
        ) : (
          <div className="bg-white border border-purple-200 rounded-lg p-8 text-center">
            <Brain className="w-8 h-8 text-gray-400 mx-auto mb-2" />
            <p className="text-sm text-gray-600">Generate BOQ, WBS, or BOM to see predictions</p>
          </div>
        )}
      </div>

      {/* BOQ Generator CTA */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold mb-2">Create Your BOQ in Minutes</h2>
            <p className="text-blue-100 mb-4">
              Use AI to extract BOQ items from text or manually add items with rate predictions
            </p>
            <button 
              onClick={() => onNavigate('boq-generator')}
              className="px-6 py-3 bg-white text-blue-600 font-semibold rounded-lg hover:bg-blue-50 transition-colors flex items-center gap-2"
            >
              <FileText className="w-5 h-5" />
              Open BOQ Generator
            </button>
          </div>
          <div className="hidden md:block">
            <FileText className="w-24 h-24 text-blue-300 opacity-50" />
          </div>
        </div>
      </div>

      {/* Projects Overview */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900">Project Portfolio</h2>
          <button className="flex items-center gap-2 px-3 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
            <Plus className="w-4 h-4" />
            New Project
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Project</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Status</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Progress</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Schedule</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Budget</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {projects.map(project => (
                <tr key={project.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <div className="font-medium text-gray-900">{project.name}</div>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium text-white ${statusColors[project.status]}`}>
                      <div className="w-1.5 h-1.5 bg-white rounded-full"></div>
                      {project.status.replace('-', ' ')}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div className="bg-blue-600 h-2 rounded-full" style={{width: `${project.progress}%`}}></div>
                      </div>
                      <span className="text-sm text-gray-600 w-10">{project.progress}%</span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`text-sm ${project.delay > 0 ? 'text-red-600' : 'text-green-600'}`}>
                      {project.delay > 0 ? `+${project.delay} days` : 'On time'}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`text-sm ${project.budget > 100 ? 'text-red-600' : 'text-gray-900'}`}>
                      {project.budget}%
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <button className="text-blue-600 hover:text-blue-800 text-sm font-medium">
                      View Details
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Bottom Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Upcoming Milestones */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h3 className="font-semibold text-gray-900 mb-3">Upcoming Milestones</h3>
          <div className="space-y-3">
            {[
              { project: 'TechCorp Office', milestone: 'Electrical Installation Complete', due: '2 days', status: 'on-track' },
              { project: 'Sharma Villa', milestone: 'Plumbing Sign-off', due: '5 days', status: 'at-risk' },
              { project: 'Factory Tank A', milestone: 'Final QC Inspection', due: '1 day', status: 'on-track' }
            ].map((item, idx) => (
              <div key={idx} className="flex items-start gap-3 p-3 border border-gray-100 rounded hover:bg-gray-50">
                <Calendar className="w-5 h-5 text-gray-400 mt-0.5" />
                <div className="flex-1">
                  <div className="font-medium text-sm text-gray-900">{item.milestone}</div>
                  <div className="text-xs text-gray-600 mt-0.5">{item.project}</div>
                  <div className="text-xs text-gray-500 mt-1">Due in {item.due}</div>
                </div>
                <span className={`px-2 py-0.5 rounded text-xs ${item.status === 'on-track' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
                  {item.status}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h3 className="font-semibold text-gray-900 mb-3">Quick Actions</h3>
          <div className="grid grid-cols-2 gap-3">
            <QuickActionCard 
              icon={FileText} 
              label="Create BOQ" 
              color="blue" 
              onClick={() => onNavigate('boq-generator')}
            />
            <QuickActionCard 
              icon={ShoppingCart} 
              label="New PO" 
              color="green" 
            />
            <QuickActionCard 
              icon={Activity} 
              label="Site Update" 
              color="purple" 
            />
            <QuickActionCard 
              icon={FileBarChart} 
              label="View Reports" 
              color="orange" 
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

