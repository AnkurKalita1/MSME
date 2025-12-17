import { useState, useEffect } from 'react';
import { Plus, FileText, Upload, AlertCircle, CheckCircle, Loader2, X, Edit2, Trash2, ArrowRight } from 'lucide-react';
import { boqAPI } from '../services/api';

const BOQGenerator = ({ onNavigate }) => {
  const [boqItems, setBoqItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [textInput, setTextInput] = useState('');
  const [showTextInput, setShowTextInput] = useState(false);
  const [editingIndex, setEditingIndex] = useState(null);
  const [editForm, setEditForm] = useState({ material: '', qty: '', unit: 'NOS', rate: '' });
  const [summary, setSummary] = useState(null);

  // Load BOQ items from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('boqItems');
    if (saved) {
      try {
        setBoqItems(JSON.parse(saved));
      } catch (e) {
        console.error('Failed to load BOQ items:', e);
      }
    }
  }, []);

  // Save BOQ items to localStorage whenever they change
  useEffect(() => {
    if (boqItems.length > 0) {
      localStorage.setItem('boqItems', JSON.stringify(boqItems));
    }
  }, [boqItems]);

  // Add new BOQ item manually
  const handleAddItem = () => {
    const newItem = {
      material: '',
      qty: 1,
      unit: 'NOS',
      rate_most_likely: 0,
      rate_min: 0,
      rate_max: 0,
      confidence: 0.75
    };
    setBoqItems([...boqItems, newItem]);
    setEditingIndex(boqItems.length);
    setEditForm({ material: '', qty: '', unit: 'NOS', rate: '' });
  };

  // Save edited item
  const handleSaveEdit = (index) => {
    // Validate material name is not empty
    if (!editForm.material || !editForm.material.trim()) {
      setError('Material name is required');
      return;
    }

    const updatedItems = [...boqItems];
    const qty = parseFloat(editForm.qty) || 0;
    const rate = parseFloat(editForm.rate) || 0;
    
    updatedItems[index] = {
      ...updatedItems[index],
      material: editForm.material.trim(),
      qty: qty,
      unit: editForm.unit || 'NOS',
      rate_most_likely: rate,
      rate_min: rate * 0.9,
      rate_max: rate * 1.1,
      confidence: updatedItems[index].confidence || 0.75
    };
    
    setBoqItems(updatedItems);
    setEditingIndex(null);
    setEditForm({ material: '', qty: '', unit: 'NOS', rate: '' });
    setError(null);
  };

  // Delete item
  const handleDeleteItem = (index) => {
    setBoqItems(boqItems.filter((_, i) => i !== index));
  };

  // Extract BOQ from text
  const handleExtractFromText = async () => {
    if (!textInput.trim()) {
      setError('Please enter text to extract BOQ from');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await boqAPI.extractAndPredict(textInput);
      
      if (response.success && response.data && response.data.items) {
        const extractedItems = response.data.items.map(item => ({
          material: item.material || '',
          qty: item.qty || 1,
          unit: item.unit || 'NOS',
          rate_most_likely: item.rate_most_likely || 0,
          rate_min: item.rate_min || 0,
          rate_max: item.rate_max || 0,
          confidence: item.confidence || 0.75,
          matched_material: item.matched_material || item.material || ''
        }));
        
        setBoqItems([...boqItems, ...extractedItems]);
        if (response.data.summary) {
          setSummary(response.data.summary);
        }
        setSuccess(`Successfully extracted ${extractedItems.length} items from text`);
        setTextInput('');
        setShowTextInput(false);
      } else {
        setError('Failed to extract BOQ items. Please try again.');
      }
    } catch (err) {
      setError(err.message || 'Failed to extract BOQ from text');
    } finally {
      setLoading(false);
    }
  };

  // Predict rates for existing items
  const handlePredictRates = async () => {
    if (boqItems.length === 0) {
      setError('Please add BOQ items first');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      // Prepare items for API - backend expects { items: [{ material: string, qty?: number }] }
      const itemsForAPI = boqItems.map(item => ({
        material: item.material || '',
        qty: item.qty || 1
      }));

      const response = await boqAPI.predictRates(itemsForAPI);
      
      if (response.success && response.data && response.data.items) {
        const updatedItems = boqItems.map((item, index) => {
          const prediction = response.data.items[index];
          if (prediction) {
            return {
              ...item,
              material: prediction.material || item.material,
              qty: prediction.qty || item.qty,
              rate_most_likely: prediction.rate_most_likely || 0,
              rate_min: prediction.rate_min || 0,
              rate_max: prediction.rate_max || 0,
              confidence: prediction.confidence || 0.75,
              matched_material: prediction.matched_material || item.material
            };
          }
          return item;
        });
        
        setBoqItems(updatedItems);
        if (response.data.summary) {
          setSummary(response.data.summary);
        }
        setSuccess('Rates predicted successfully');
      } else {
        setError('Failed to predict rates. Please try again.');
      }
    } catch (err) {
      setError(err.message || 'Failed to predict rates');
    } finally {
      setLoading(false);
    }
  };

  // Calculate total (using rate_most_likely * qty)
  const totalAmount = boqItems.reduce((sum, item) => sum + ((item.qty || 0) * (item.rate_most_likely || 0)), 0);

  // Navigate to WBS Creator with BOQ data
  const handleGenerateWBS = () => {
    if (boqItems.length === 0) {
      setError('Please add at least one BOQ item before generating WBS');
      return;
    }

    // If there's an item being edited, check if it has valid data and save it
    let itemsToValidate = [...boqItems];
    if (editingIndex !== null && editingIndex >= 0 && editingIndex < itemsToValidate.length) {
      if (editForm.material && editForm.material.trim()) {
        // Auto-save the item being edited
        const qty = parseFloat(editForm.qty) || 0;
        const rate = parseFloat(editForm.rate) || 0;
        
        itemsToValidate[editingIndex] = {
          ...itemsToValidate[editingIndex],
          material: editForm.material.trim(),
          qty: qty,
          rate_most_likely: rate,
          rate_min: rate * 0.9,
          rate_max: rate * 1.1,
          confidence: itemsToValidate[editingIndex].confidence || 0.75
        };
        
        // Update the state immediately
        setBoqItems(itemsToValidate);
        setEditingIndex(null);
        setEditForm({ material: '', qty: '', rate: '' });
      } else {
        setError('Please enter a material name for the item you\'re editing');
        return;
      }
    }

    // Filter out items without material names (only use valid items)
    const validItems = itemsToValidate.filter(item => item.material && item.material.trim());
    
    if (validItems.length === 0) {
      setError('Please ensure at least one item has a material name');
      return;
    }

    // Show warning if some items were filtered out
    if (validItems.length < itemsToValidate.length) {
      const filteredCount = itemsToValidate.length - validItems.length;
      setSuccess(`Generating WBS for ${validItems.length} valid item(s). ${filteredCount} item(s) without material names were skipped.`);
    } else {
      setSuccess('Generating WBS...');
    }

    // Save only valid BOQ items to localStorage for WBS page
    localStorage.setItem('boqItems', JSON.stringify(validItems));
    
    // Navigate to WBS Creator
    if (onNavigate) {
      onNavigate('wbs-creator');
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Smart BOQ Generator</h1>
          <p className="text-gray-600 mt-1">Create Bill of Quantities with AI assistance</p>
        </div>
        <button 
          onClick={handleAddItem}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          <Plus className="w-4 h-4" />
          Add Item
        </button>
      </div>

      {/* AI Upload Section */}
      <div className="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg p-6">
        <div className="flex items-start gap-4">
          <div className="bg-purple-500 rounded-lg p-3">
            <FileText className="w-6 h-6 text-white" />
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-gray-900 mb-2">AI-Powered BOQ Creation</h3>
            <p className="text-sm text-gray-700 mb-4">
              Upload client requirements or enter text to extract items and suggest pricing
            </p>
            <div className="flex gap-3 flex-wrap">
              <button 
                onClick={() => setShowTextInput(!showTextInput)}
                className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
              >
                <Upload className="w-4 h-4 inline mr-2" />
                Extract from Text
              </button>
              <button 
                onClick={handlePredictRates}
                disabled={loading || boqItems.length === 0}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 inline mr-2 animate-spin" />
                    Predicting...
                  </>
                ) : (
                  'Predict Rates'
                )}
              </button>
            </div>

            {/* Text Input Area */}
            {showTextInput && (
              <div className="mt-4 space-y-2">
                <textarea
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  placeholder="Enter project requirements, BOQ text, or description here..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  rows="4"
                />
                <div className="flex gap-2">
                  <button
                    onClick={handleExtractFromText}
                    disabled={loading || !textInput.trim()}
                    className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="w-4 h-4 inline mr-2 animate-spin" />
                        Extracting...
                      </>
                    ) : (
                      'Extract BOQ'
                    )}
                  </button>
                  <button
                    onClick={() => {
                      setShowTextInput(false);
                      setTextInput('');
                    }}
                    className="px-4 py-2 border border-gray-300 text-gray-700 rounded hover:bg-gray-50"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Error/Success Messages */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm text-red-800">{error}</p>
          </div>
          <button onClick={() => setError(null)}>
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

      {/* Summary Section */}
      {summary && (
        <div className="bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg p-4">
          <h3 className="font-semibold text-gray-900 mb-3">Project Summary</h3>
          <div className="grid grid-cols-4 gap-4">
            <div>
              <div className="text-sm text-gray-600">Total Items</div>
              <div className="text-lg font-bold text-gray-900">{summary.total_items || boqItems.length}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Min Cost</div>
              <div className="text-lg font-bold text-gray-900">₹{summary.project_cost_min?.toLocaleString('en-IN') || '0'}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Most Likely Cost</div>
              <div className="text-lg font-bold text-blue-600">₹{summary.project_cost_most_likely?.toLocaleString('en-IN') || totalAmount.toLocaleString('en-IN')}</div>
            </div>
            <div>
              <div className="text-sm text-gray-600">Max Cost</div>
              <div className="text-lg font-bold text-gray-900">₹{summary.project_cost_max?.toLocaleString('en-IN') || '0'}</div>
            </div>
          </div>
        </div>
      )}

      {/* BOQ Table */}
      {boqItems.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200">
          <div className="p-4 border-b border-gray-200 flex items-center justify-between">
            <h2 className="font-semibold text-gray-900">BOQ Items ({boqItems.length})</h2>
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600">Total: ₹{totalAmount.toLocaleString('en-IN')}</span>
              <button 
                onClick={handleGenerateWBS}
                className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 flex items-center gap-2"
              >
                Generate WBS
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Material</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Quantity (qty)</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Unit</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Rate (rate_most_likely)</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Rate Range</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Total Cost</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Confidence</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {boqItems.map((item, idx) => (
                  <tr key={idx} className="hover:bg-gray-50">
                    {editingIndex === idx ? (
                      <>
                        <td className="px-4 py-3">
                          <input
                            type="text"
                            value={editForm.material}
                            onChange={(e) => setEditForm({ ...editForm, material: e.target.value })}
                            className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                            placeholder="Material name"
                          />
                        </td>
                        <td className="px-4 py-3">
                          <input
                            type="number"
                            value={editForm.qty}
                            onChange={(e) => setEditForm({ ...editForm, qty: e.target.value })}
                            className="w-20 px-2 py-1 border border-gray-300 rounded text-sm"
                            placeholder="Qty"
                          />
                        </td>
                        <td className="px-4 py-3">
                          <select
                            value={editForm.unit || 'NOS'}
                            onChange={(e) => setEditForm({ ...editForm, unit: e.target.value })}
                            className="px-2 py-1 border border-gray-300 rounded text-sm"
                          >
                            {/* Common BOQ units, aligned with backend datasets */}
                            <option value="NOS">NOS</option>
                            <option value="Sqft">Sqft</option>
                            <option value="Sqm">Sqm</option>
                            <option value="Cft">Cft</option>
                            <option value="Rft">Rft</option>
                            <option value="Mtr">Mtr</option>
                            <option value="Kg">Kg</option>
                            <option value="Ltr">Ltr</option>
                            <option value="Ton">Ton</option>
                            <option value="Pcs">Pcs</option>
                            <option value="Point">Point</option>
                          </select>
                        </td>
                        <td className="px-4 py-3">
                          <input
                            type="number"
                            value={editForm.rate}
                            onChange={(e) => setEditForm({ ...editForm, rate: e.target.value })}
                            className="w-24 px-2 py-1 border border-gray-300 rounded text-sm"
                            placeholder="Rate"
                          />
                        </td>
                        <td className="px-4 py-3 text-xs text-gray-500">
                          Min: ₹{((parseFloat(editForm.rate) || 0) * 0.9).toFixed(2)}<br/>
                          Max: ₹{((parseFloat(editForm.rate) || 0) * 1.1).toFixed(2)}
                        </td>
                        <td className="px-4 py-3 font-medium text-gray-900">
                          ₹{((parseFloat(editForm.qty) || 0) * (parseFloat(editForm.rate) || 0)).toLocaleString('en-IN')}
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-1 rounded-full text-xs ${
                            (item.confidence || 0) >= 0.7 ? 'bg-green-100 text-green-700' : 
                            (item.confidence || 0) >= 0.5 ? 'bg-yellow-100 text-yellow-700' : 
                            'bg-gray-100 text-gray-700'
                          }`}>
                            {((item.confidence || 0) * 100).toFixed(0)}%
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex gap-2">
                            <button
                              onClick={() => handleSaveEdit(idx)}
                              className="text-green-600 hover:text-green-800 text-sm"
                            >
                              Save
                            </button>
                            <button
                              onClick={() => {
                                setEditingIndex(null);
                                setEditForm({ material: '', qty: '', unit: 'NOS', rate: '' });
                              }}
                              className="text-gray-600 hover:text-gray-800 text-sm"
                            >
                              Cancel
                            </button>
                          </div>
                        </td>
                      </>
                    ) : (
                      <>
                        <td className="px-4 py-3">
                          {!item.material || !item.material.trim() ? (
                            <div className="flex items-center gap-2">
                              <span className="text-red-500 text-xs">⚠️</span>
                              <span className="text-red-600 italic text-sm">Material name required</span>
                            </div>
                          ) : (
                            <>
                              <div className="font-medium text-gray-900">{item.material}</div>
                              {item.matched_material && item.matched_material !== item.material && (
                                <div className="text-xs text-gray-500">Matched: {item.matched_material}</div>
                              )}
                            </>
                          )}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-600">{item.qty || 0}</td>
                        <td className="px-4 py-3 text-sm text-gray-600">{item.unit || 'NOS'}</td>
                        <td className="px-4 py-3 text-sm text-gray-600">
                          ₹{item.rate_most_likely?.toLocaleString('en-IN') || '0'}
                        </td>
                        <td className="px-4 py-3 text-xs text-gray-500">
                          {item.rate_min && item.rate_max && (
                            <>
                              Min: ₹{item.rate_min.toLocaleString('en-IN')}<br/>
                              Max: ₹{item.rate_max.toLocaleString('en-IN')}
                            </>
                          )}
                        </td>
                        <td className="px-4 py-3 font-medium text-gray-900">
                          ₹{((item.qty || 0) * (item.rate_most_likely || 0)).toLocaleString('en-IN')}
                        </td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-1 rounded-full text-xs ${
                            (item.confidence || 0) >= 0.7 ? 'bg-green-100 text-green-700' : 
                            (item.confidence || 0) >= 0.5 ? 'bg-yellow-100 text-yellow-700' : 
                            'bg-gray-100 text-gray-700'
                          }`}>
                            {((item.confidence || 0) * 100).toFixed(0)}%
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex gap-2">
                            <button
                              onClick={() => {
                                setEditingIndex(idx);
                                setEditForm({
                                  material: item.material || '',
                                  qty: item.qty || 1,
                                  unit: item.unit || 'NOS',
                                  rate: item.rate_most_likely || 0
                                });
                              }}
                              className="text-blue-600 hover:text-blue-800 text-sm"
                            >
                              <Edit2 className="w-4 h-4 inline" />
                            </button>
                            <button
                              onClick={() => handleDeleteItem(idx)}
                              className="text-red-600 hover:text-red-800 text-sm"
                            >
                              <Trash2 className="w-4 h-4 inline" />
                            </button>
                          </div>
                        </td>
                      </>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {boqItems.length === 0 && !loading && (
        <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
          <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">No BOQ items yet. Add items manually or extract from text.</p>
        </div>
      )}
    </div>
  );
};

export default BOQGenerator;

