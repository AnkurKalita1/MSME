import { useState, useEffect } from 'react';
import { Plus, FileText, Loader2, AlertCircle, CheckCircle, X, ChevronDown, ChevronRight, Calendar, Clock, User, ShoppingCart, ArrowRight } from 'lucide-react';
import { wbsAPI } from '../services/api';

const WBSCreator = ({ onNavigate }) => {
  const [wbsData, setWbsData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [boqInput, setBoqInput] = useState({
    boq_text: '',
    quantity: 1,
    region: 'central',
    season: 'normal',
    grade: 'B'
  });
  const [selectedBoqIndex, setSelectedBoqIndex] = useState(null);
  const [expandedItems, setExpandedItems] = useState({});
  // Track which stage is currently selected for each WBS card (for filtering tasks)
  // key: wbs index, value: stage name or 'All'
  const [stageFilters, setStageFilters] = useState({});
  // Track which task is selected to show detailed info (per WBS card)
  // shape: { wbsIndex: number, taskIndex: number } | null
  const [selectedTask, setSelectedTask] = useState(null);
  const [boqItemsFromStorage, setBoqItemsFromStorage] = useState([]);

  // Load BOQ items from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('boqItems');
    if (saved) {
      try {
        const items = JSON.parse(saved);
        setBoqItemsFromStorage(items);
        // Pre-fill first item if available
        if (items.length > 0) {
          setBoqInput({
            boq_text: items[0].material || '',
            quantity: items[0].qty || 1,
            region: 'central',
            season: 'normal',
            grade: 'B'
          });
          setSelectedBoqIndex(0);
        }
      } catch (e) {
        console.error('Failed to load BOQ items:', e);
      }
    }
  }, []);

  // Handle selecting a BOQ item from the list
  const handleSelectBoqItem = (item, index) => {
    setBoqInput({
      boq_text: item.material || '',
      quantity: item.qty || 1,
      region: 'central',
      season: 'normal',
      grade: 'B'
    });
    setSelectedBoqIndex(index);
  };

  // Toggle expand/collapse for WBS item
  const toggleExpand = (index) => {
    setExpandedItems({
      ...expandedItems,
      [index]: !expandedItems[index]
    });

    // When first expanding a card, default the stage filter to 'All'
    if (!expandedItems[index]) {
      setStageFilters((prev) => ({
        ...prev,
        [index]: prev[index] || 'All'
      }));
    }
  };

  // Handle clicking on a stage pill inside a WBS card
  const handleStageClick = (wbsIndex, stageName) => {
    setStageFilters((prev) => ({
      ...prev,
      [wbsIndex]: stageName
    }));
  };

  // Handle clicking on a specific task to show more details
  const handleTaskClick = (wbsIndex, taskIndex) => {
    setSelectedTask((prev) => {
      if (prev && prev.wbsIndex === wbsIndex && prev.taskIndex === taskIndex) {
        // Clicking again on the same task collapses the details
        return null;
      }
      return { wbsIndex, taskIndex };
    });
  };

  // Generate WBS from single BOQ item
  const handleGenerateWBS = async () => {
    if (!boqInput.boq_text.trim()) {
      setError('Please enter BOQ item description');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      // Find matching BOQ item from storage to get rate data
      const matchingBoqItem = boqItemsFromStorage.find(item => 
        item.material === boqInput.boq_text
      );

      const payload = {
        boq_text: boqInput.boq_text,
        quantity: parseFloat(boqInput.quantity) || 1.0,
        region: boqInput.region,
        season: boqInput.season,
        grade: boqInput.grade,
        rate_most: matchingBoqItem?.rate_most_likely || 0.0,
        rate_min: matchingBoqItem?.rate_min || 0.0,
        rate_max: matchingBoqItem?.rate_max || 0.0,
        confidence: matchingBoqItem?.confidence || 1.0
      };

      const response = await wbsAPI.generate(payload);
      
      if (response.success && response.data) {
        // Extract tasks from wbs object structure
        const wbsObj = response.data.wbs || {};
        const allTasks = [];
        const stageNames = [];
        
        // Process each stage
        Object.keys(wbsObj).forEach(stageName => {
          if (Array.isArray(wbsObj[stageName])) {
            stageNames.push(stageName);
            wbsObj[stageName].forEach(task => {
              allTasks.push({
                ...task,
                stage: stageName
              });
            });
          }
        });

        // Calculate total duration from all tasks
        const totalDuration = allTasks.reduce((sum, task) => {
          const duration = task.duration || {};
          return sum + (duration.expected_hours || duration.most_likely_hours || 0);
        }, 0) / 8; // Convert hours to days

        const newWBS = {
          boqItem: response.data.boq_text || boqInput.boq_text,
          quantity: boqInput.quantity,
          tasks: allTasks,
          stages: stageNames,
          totalDuration: Math.round(totalDuration * 10) / 10,
          criticalPath: [],
          wbs: wbsObj,
          subcategory: response.data.subcategory || '',
          wbsId: response.data.wbs_id || ''
        };
        
        setWbsData([...wbsData, newWBS]);
        setSuccess('WBS generated successfully');
        setBoqInput({
          boq_text: '',
          quantity: 1,
          region: 'central',
          season: 'normal',
          grade: 'B'
        });
      } else {
        setError('Failed to generate WBS. Please try again.');
      }
    } catch (err) {
      setError(err.message || 'Failed to generate WBS');
    } finally {
      setLoading(false);
    }
  };

  // Generate WBS for multiple BOQ items (batch)
  const handleGenerateBatchWBS = async (boqItems) => {
    if (!boqItems || boqItems.length === 0) {
      setError('Please provide BOQ items');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const boqs = boqItems.map(item => ({
        boq_text: item.material || item.boq_text || item.item || '',
        quantity: item.qty || item.quantity || 1.0,
        region: item.region || 'central',
        season: item.season || 'normal',
        grade: item.grade || 'B',
        rate_most: item.rate_most_likely || item.rate_most || 0.0,
        rate_min: item.rate_min || 0.0,
        rate_max: item.rate_max || 0.0,
        confidence: item.confidence || 1.0
      }));

      const response = await wbsAPI.generateBatch(boqs);
      
      // Debug logging
      console.log('Batch WBS Response:', response);
      console.log('BOQs sent:', boqs);
      
      // Backend returns { success: true, results: [...] } directly
      if (response && response.success && response.results && Array.isArray(response.results)) {
        const newWBSItems = response.results.map((result, index) => {
          // Debug each result
          console.log(`Result ${index}:`, result);
          
          if (result.success && result.data) {
            // Extract tasks from all stages in the wbs object
            const wbsObj = result.data.wbs || {};
            const allTasks = [];
            const stageNames = [];
            
            // Debug wbs structure
            console.log(`WBS Object for item ${index}:`, wbsObj);
            console.log(`Full result.data for item ${index}:`, result.data);
            
            // Check if wbs object exists and has content
            if (!wbsObj || Object.keys(wbsObj).length === 0) {
              console.warn(`Empty or missing WBS object for item ${index}. Full data:`, result.data);
              // Still create a WBS item even if empty, so user can see something was processed
              return {
                boqItem: result.data.boq_text || boqs[index].boq_text,
                quantity: boqs[index].quantity,
                tasks: [],
                stages: [],
                totalDuration: 0,
                criticalPath: [],
                wbs: {},
                subcategory: result.data.subcategory || '',
                wbsId: result.data.wbs_id || '',
                error: 'WBS object is empty or missing'
              };
            }
            
            // Process each stage
            Object.keys(wbsObj).forEach(stageName => {
              if (Array.isArray(wbsObj[stageName])) {
                stageNames.push(stageName);
                wbsObj[stageName].forEach(task => {
                  allTasks.push({
                    ...task,
                    stage: stageName
                  });
                });
              }
            });

            // If no tasks found, log for debugging but still return the item
            if (allTasks.length === 0) {
              console.warn(`No tasks found in WBS for item ${index}. WBS structure:`, wbsObj);
            }

            // Calculate total duration from all tasks
            const totalDuration = allTasks.reduce((sum, task) => {
              const duration = task.duration || {};
              return sum + (duration.expected_hours || duration.most_likely_hours || 0);
            }, 0) / 8; // Convert hours to days (assuming 8 hours per day)

            return {
              boqItem: result.data.boq_text || boqs[index].boq_text,
              quantity: boqs[index].quantity,
              tasks: allTasks,
              stages: stageNames,
              totalDuration: Math.round(totalDuration * 10) / 10, // Round to 1 decimal
              criticalPath: [], // Can be calculated if needed
              wbs: wbsObj,
              subcategory: result.data.subcategory || '',
              wbsId: result.data.wbs_id || ''
            };
          } else {
            // Log failed results
            console.warn(`Result ${index} failed:`, result);
            if (result.error) {
              console.error(`Error for item ${index}:`, result.error);
            }
          }
          return null;
        }).filter(item => item !== null);
        
        console.log('Processed WBS Items:', newWBSItems);
        
        if (newWBSItems.length > 0) {
          setWbsData([...wbsData, ...newWBSItems]);
          setSuccess(`Successfully generated WBS for ${newWBSItems.length} item(s)`);
        } else {
          // More detailed error message
          const failedCount = response.results.filter(r => !r.success || !r.data).length;
          const errorDetails = response.results
            .map((r, i) => r.error ? `Item ${i + 1}: ${r.error}` : null)
            .filter(Boolean)
            .join('; ');
          
          // Check for Python path errors
          const hasPythonError = errorDetails && errorDetails.includes('Python process');
          let errorMessage = `WBS generation completed but no valid items were returned. ${failedCount} item(s) failed.`;
          
          if (hasPythonError) {
            errorMessage += `\n\n⚠️ Backend Configuration Issue: Python executable not found. ` +
              `Please ensure Python 3 is installed and set the PYTHON_PATH environment variable in the backend. ` +
              `On macOS/Linux, try: export PYTHON_PATH=python3 before starting the backend server.`;
          } else if (errorDetails) {
            errorMessage += `\n\nErrors: ${errorDetails}`;
          } else {
            errorMessage += `\n\nPlease check the backend response.`;
          }
          
          setError(errorMessage);
        }
      } else {
        const responseStr = response ? JSON.stringify(response).substring(0, 200) : 'null';
        setError(`Failed to generate batch WBS. Response: ${responseStr}. Expected: { success: true, results: [...] }`);
      }
    } catch (err) {
      setError(err.message || 'Failed to generate batch WBS');
    } finally {
      setLoading(false);
    }
  };

  // Get stage color
  const getStageColor = (stage) => {
    const colors = {
      'Planning': 'bg-blue-100 text-blue-700',
      'Procurement': 'bg-purple-100 text-purple-700',
      'Execution': 'bg-green-100 text-green-700',
      'Quality Control': 'bg-yellow-100 text-yellow-700',
      'Billing & Handover': 'bg-orange-100 text-orange-700'
    };
    return colors[stage] || 'bg-gray-100 text-gray-700';
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">AI-Driven WBS Creator</h1>
          <p className="text-gray-600 mt-1">Generate Work Breakdown Schedule from BOQ items</p>
        </div>
      </div>

      {/* Input Form */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h2 className="font-semibold text-gray-900 mb-4">Generate WBS from BOQ Item</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              BOQ Item Description *
            </label>
            <input
              type="text"
              value={boqInput.boq_text}
              onChange={(e) => setBoqInput({ ...boqInput, boq_text: e.target.value })}
              placeholder="e.g., Modular Workstations (6-seater)"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Quantity
            </label>
            <input
              type="number"
              value={boqInput.quantity}
              onChange={(e) => setBoqInput({ ...boqInput, quantity: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Region
            </label>
            <select
              value={boqInput.region}
              onChange={(e) => setBoqInput({ ...boqInput, region: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="central">Central</option>
              <option value="north">North</option>
              <option value="south">South</option>
              <option value="east">East</option>
              <option value="west">West</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Season
            </label>
            <select
              value={boqInput.season}
              onChange={(e) => setBoqInput({ ...boqInput, season: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="normal">Normal</option>
              <option value="monsoon">Monsoon</option>
              <option value="summer">Summer</option>
              <option value="winter">Winter</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Grade
            </label>
            <select
              value={boqInput.grade}
              onChange={(e) => setBoqInput({ ...boqInput, grade: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="A">Grade A</option>
              <option value="B">Grade B</option>
              <option value="C">Grade C</option>
            </select>
          </div>
        </div>
        <div className="mt-4 flex gap-3">
          <button
            onClick={handleGenerateWBS}
            disabled={loading || !boqInput.boq_text.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 inline mr-2 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Plus className="w-4 h-4 inline mr-2" />
                Generate WBS
              </>
            )}
          </button>
          {boqItemsFromStorage.length > 0 && (
            <button
              onClick={() => handleGenerateBatchWBS(boqItemsFromStorage)}
              disabled={loading}
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 inline mr-2 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  Generate WBS for All BOQ Items ({boqItemsFromStorage.length})
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {/* BOQ Items from Storage */}
      {boqItemsFromStorage.length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-gray-900 mb-2">BOQ Items from Generator ({boqItemsFromStorage.length})</h3>
          <p className="text-sm text-gray-700 mb-3">
            Click on any item below to use it in the form above. You can also generate WBS for all items at once using the button above.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {boqItemsFromStorage.map((item, idx) => (
              <div 
                key={idx} 
                onClick={() => handleSelectBoqItem(item, idx)}
                className={`bg-white p-3 rounded border-2 cursor-pointer transition-all hover:shadow-md text-sm ${
                  selectedBoqIndex === idx 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-blue-100 hover:border-blue-300'
                }`}
              >
                <div className="font-medium text-gray-900">{item.material}</div>
                <div className="text-xs text-gray-600 mt-1">
                  Qty: {item.qty} {item.unit || 'NOS'} | Rate: ₹{item.rate_most_likely || 0}
                </div>
                {selectedBoqIndex === idx && (
                  <div className="text-xs text-blue-600 font-medium mt-1">✓ Selected</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error/Success Messages */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
          <div className="flex-1">
            <div className="text-sm text-red-800 whitespace-pre-line">{error}</div>
          </div>
          <button onClick={() => setError(null)} className="flex-shrink-0">
            <X className="w-4 h-4 text-red-600" />
          </button>
        </div>
      )}

      {success && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-start gap-3">
          <CheckCircle className="w-5 h-5 text-green-600 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm text-green-800">{success}</p>
          </div>
          <button onClick={() => setSuccess(null)}>
            <X className="w-4 h-4 text-green-600" />
          </button>
        </div>
      )}

      {/* Generate BOM Button - Only visible after successful WBS generation */}
      {wbsData.length > 0 && (
        <div className="bg-gradient-to-r from-purple-50 to-indigo-50 border border-purple-200 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-purple-500 rounded-lg p-2">
                <ShoppingCart className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">WBS Generated Successfully</h3>
                <p className="text-sm text-gray-700 mt-1">
                  {wbsData.length} WBS item(s) ready. Proceed to generate Bill of Materials (BOM).
                </p>
              </div>
            </div>
            <button
              onClick={() => onNavigate && onNavigate('bom-generator')}
              className="flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 font-medium shadow-sm transition-colors"
            >
              <ShoppingCart className="w-5 h-5" />
              Generate BOM
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* WBS Results */}
      {wbsData.length > 0 && (
        <div className="space-y-4">
          {wbsData.map((wbs, index) => (
            <div key={index} className="bg-white rounded-lg border border-gray-200">
              <div className="p-4 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <h3 className="font-semibold text-gray-900">{wbs.boqItem}</h3>
                    <div className="flex items-center gap-4 mt-2 text-sm text-gray-600">
                      <span>Quantity: {wbs.quantity}</span>
                      {wbs.totalDuration > 0 && (
                        <span className="flex items-center gap-1">
                          <Clock className="w-4 h-4" />
                          Total Duration: {wbs.totalDuration} days
                        </span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => toggleExpand(index)}
                    className="p-2 hover:bg-gray-100 rounded"
                  >
                    {expandedItems[index] ? (
                      <ChevronDown className="w-5 h-5 text-gray-600" />
                    ) : (
                      <ChevronRight className="w-5 h-5 text-gray-600" />
                    )}
                  </button>
                </div>
              </div>

              {expandedItems[index] && (
                <div className="p-4 space-y-4">
                  {/* Stages */}
                  {wbs.stages && wbs.stages.length > 0 && (
                    <div>
                      <h4 className="font-medium text-gray-900 mb-3">Stages</h4>
                      <div className="flex flex-wrap gap-2">
                        {/* 'All' stage filter */}
                        <button
                          type="button"
                          onClick={() => handleStageClick(index, 'All')}
                          className={`px-3 py-1 rounded-full text-xs font-medium border ${
                            (stageFilters[index] || 'All') === 'All'
                              ? 'bg-blue-600 text-white border-blue-600'
                              : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                          }`}
                        >
                          All stages
                        </button>

                        {wbs.stages.map((stage, stageIdx) => {
                          const stageKey = stage.name || stage;
                          const isActive = stageFilters[index] === stageKey;
                          return (
                            <button
                              type="button"
                              key={stageIdx}
                              onClick={() => handleStageClick(index, stageKey)}
                              className={`px-3 py-1 rounded-full text-xs font-medium border ${
                                isActive
                                  ? `${getStageColor(stageKey)} border-transparent`
                                  : `bg-white text-gray-700 border-gray-300 hover:bg-gray-50`
                              }`}
                            >
                              {stageKey}
                            </button>
                          );
                        })}
                      </div>
                      <p className="mt-2 text-xs text-gray-500">
                        Click on a stage pill to filter the tasks below.{' '}
                        <span className="font-medium">
                          Showing:{' '}
                          {(stageFilters[index] || 'All') === 'All'
                            ? 'All stages'
                            : stageFilters[index]}
                        </span>
                      </p>
                    </div>
                  )}

                  {/* Tasks */}
                  {wbs.tasks && wbs.tasks.length > 0 && (
                    <div>
                      <h4 className="font-medium text-gray-900 mb-3">Tasks</h4>
                      <div className="space-y-2">
                        {wbs.tasks
                          .filter((task) => {
                            const activeStage = stageFilters[index] || 'All';
                            if (activeStage === 'All') return true;
                            // task.stage holds the stage name we added when flattening
                            return (task.stage || '').toLowerCase() === activeStage.toLowerCase();
                          })
                          .map((task, taskIdx) => (
                          <div
                            key={taskIdx}
                            onClick={() => handleTaskClick(index, taskIdx)}
                            className={`border rounded-lg p-3 cursor-pointer transition-all ${
                              selectedTask &&
                              selectedTask.wbsIndex === index &&
                              selectedTask.taskIndex === taskIdx
                                ? 'border-blue-500 bg-blue-50 shadow-sm'
                                : 'border-gray-200 hover:border-blue-300 hover:shadow-sm'
                            }`}
                          >
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                <div className="font-medium text-gray-900">
                                  {task.task_name || task.name || task.description || `Task ${taskIdx + 1}`}
                                </div>
                                {task.description && (
                                  <div className="text-sm text-gray-600 mt-1">{task.description}</div>
                                )}
                              </div>
                            </div>
                            <div className="flex items-center gap-4 mt-2 text-sm text-gray-600">
                              {task.duration && (
                                <span className="flex items-center gap-1">
                                  <Clock className="w-4 h-4" />
                                  {typeof task.duration === 'object' 
                                    ? `${(task.duration.expected_hours || task.duration.most_likely_hours || 0) / 8} days`
                                    : `${task.duration} days`}
                                  {typeof task.duration === 'object' && task.duration.most_likely_hours && (
                                    <span className="text-xs text-gray-500 ml-1">
                                      ({task.duration.most_likely_hours}h)
                                    </span>
                                  )}
                                </span>
                              )}
                              {task.assignee && (
                                <span className="flex items-center gap-1">
                                  <User className="w-4 h-4" />
                                  {task.assignee}
                                </span>
                              )}
                              {task.stage && (
                                <span className={`px-2 py-0.5 rounded text-xs ${getStageColor(task.stage)}`}>
                                  {task.stage}
                                </span>
                              )}
                            </div>
                            {task.dependencies && task.dependencies.length > 0 && (
                              <div className="mt-2 text-xs text-gray-500">
                                Dependencies: {task.dependencies.join(', ')}
                              </div>
                            )}

                            {/* Detailed data for selected task (from backend response) */}
                            {selectedTask &&
                              selectedTask.wbsIndex === index &&
                              selectedTask.taskIndex === taskIdx && (
                                <div className="mt-3 pt-3 border-t border-dashed border-gray-200 text-xs text-gray-700 space-y-2">
                                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                                    <div>
                                      <div className="text-gray-500">Task ID</div>
                                      <div className="font-mono text-gray-900">
                                        {task.task_id || 'N/A'}
                                      </div>
                                    </div>
                                    <div>
                                      <div className="text-gray-500">Stage (numeric)</div>
                                      <div className="font-mono text-gray-900">
                                        {task.stage || 'N/A'}
                                      </div>
                                    </div>
                                    {typeof task.duration === 'object' && (
                                      <>
                                        <div>
                                          <div className="text-gray-500">Optimistic hours</div>
                                          <div className="font-mono text-gray-900">
                                            {task.duration.optimistic_hours}
                                          </div>
                                        </div>
                                        <div>
                                          <div className="text-gray-500">Most likely hours</div>
                                          <div className="font-mono text-gray-900">
                                            {task.duration.most_likely_hours}
                                          </div>
                                        </div>
                                        <div>
                                          <div className="text-gray-500">Pessimistic hours</div>
                                          <div className="font-mono text-gray-900">
                                            {task.duration.pessimistic_hours}
                                          </div>
                                        </div>
                                        <div>
                                          <div className="text-gray-500">Expected hours</div>
                                          <div className="font-mono text-gray-900">
                                            {task.duration.expected_hours}
                                          </div>
                                        </div>
                                      </>
                                    )}
                                  </div>
                                </div>
                              )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Critical Path */}
                  {wbs.criticalPath && wbs.criticalPath.length > 0 && (
                    <div>
                      <h4 className="font-medium text-gray-900 mb-3">Critical Path</h4>
                      <div className="flex flex-wrap gap-2">
                        {wbs.criticalPath.map((item, cpIdx) => (
                          <span key={cpIdx} className="px-2 py-1 bg-red-100 text-red-700 rounded text-sm">
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {wbsData.length === 0 && !loading && (
        <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
          <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">No WBS generated yet. Enter a BOQ item to generate WBS.</p>
        </div>
      )}
    </div>
  );
};

export default WBSCreator;

