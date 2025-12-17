import { useState, useEffect } from 'react';
import { Plus, Package, Loader2, AlertCircle, CheckCircle, X, ChevronDown, ChevronRight, DollarSign, TrendingUp } from 'lucide-react';
import { bomAPI } from '../services/api';

const BOMGenerator = () => {
  const [bomData, setBomData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [boqInput, setBoqInput] = useState({
    category: '',
    item: '',
    quantity: 1,
    unit: 'Nos',
    // Optional context fields (aligned with backend schema)
    region: 'central',
    season: 'normal',
    grade: 'standard'
  });
  const [expandedItems, setExpandedItems] = useState({});
  const [boqItemsFromStorage, setBoqItemsFromStorage] = useState([]);
  const [selectedBoqIndex, setSelectedBoqIndex] = useState(null);

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
            category: items[0].category || '',
            item: items[0].material || '',
            quantity: items[0].qty || 1,
            unit: items[0].unit || 'Nos',
            region: 'central',
            season: 'normal',
            grade: 'standard'
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
      category: item.category || '',
      item: item.material || '',
      quantity: item.qty || 1,
      unit: item.unit || 'Nos',
      region: 'central',
      season: 'normal',
      grade: 'standard'
    });
    setSelectedBoqIndex(index);
  };

  // Toggle expand/collapse for BOM item
  const toggleExpand = (index) => {
    setExpandedItems({
      ...expandedItems,
      [index]: !expandedItems[index]
    });
  };

  // Generate BOM from BOQ item
  const handleGenerateBOM = async () => {
    if (!boqInput.category || !boqInput.item) {
      setError('Please enter category and item description');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      // Align payload with backend bom_api.py schema
      // Backend expects:
      // {
      //   "boq": {
      //     "boq_id": string?,
      //     "category": string,
      //     "material": string,
      //     "quantity": number,
      //     "unit": string,
      //     "region": string,
      //     "season": string,
      //     "grade": string,
      //     "dimensions": {...}?,
      //     "project_context": {...}?
      //   },
      //   "wbs": {
      //     "execution_tasks": [...]
      //   }
      // }
      const payload = {
        boq: {
          category: boqInput.category,
          material: boqInput.item,
          quantity: parseFloat(boqInput.quantity) || 1,
          unit: boqInput.unit,
          region: boqInput.region || 'central',
          season: boqInput.season || 'normal',
          grade: boqInput.grade || 'standard'
        },
        // We are not yet wiring WBS usage into BOM, so pass empty structure
        wbs: {
          execution_tasks: []
        }
      };

      const response = await bomAPI.generate(payload);

      // Backend controller wraps python result as: { success: true, data: { ... } }
      if (response.success && response.data) {
        const bomResult = response.data;
        const materials = bomResult.bom_lines || [];
        const totalCost = materials.reduce(
          (sum, m) => sum + (m.total_cost_inr || 0),
          0
        );

        const newBOM = {
          boqItem: boqInput.item,
          category: boqInput.category,
          quantity: boqInput.quantity,
          unit: boqInput.unit,
          materials,
          totalCost,
          // Use model validation MAPE as a proxy quality indicator (optional)
          wastageFactor: bomResult.wastage_model_val_mape || 0
        };

        setBomData([...bomData, newBOM]);
        setSuccess('BOM generated successfully');
        setBoqInput({
          category: '',
          item: '',
          quantity: 1,
          unit: 'Nos',
          region: 'central',
          season: 'normal',
          grade: 'standard'
        });
      } else {
        setError('Failed to generate BOM. Please try again.');
      }
    } catch (err) {
      setError(err.message || 'Failed to generate BOM');
    } finally {
      setLoading(false);
    }
  };

  // Generate BOM for multiple BOQ items (batch)
  const handleGenerateBatchBOM = async (boqItems) => {
    if (!boqItems || boqItems.length === 0) {
      setError('Please provide BOQ items');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      // Align batch payload with backend schema (same shape as single generate)
      // IMPORTANT: Each item uses its own category (if available), NOT the form's category
      // This ensures individual items can have different categories
      const boqs = boqItems.map(item => ({
        boq: {
          category: item.category || 'General', // Use item's own category, or default to 'General' (NOT form category)
          material: item.material || item.item || '',
          quantity: item.qty || item.quantity || 1,
          unit: item.unit || 'Nos',
          region: item.region || 'central', // Use item's own region, or default
          season: item.season || 'normal', // Use item's own season, or default
          grade: item.grade || 'standard' // Use item's own grade, or default
        },
        wbs: {
          execution_tasks: []
        }
      }));

      const response = await bomAPI.generateBatch(boqs);

      // Backend returns: { success: true, data: { results: [ { success, data }, ... ] } }
      if (response.success && response.data && Array.isArray(response.data.results)) {
        const results = response.data.results;

        const newBOMItems = results.map((result, index) => {
          if (result.success && result.data) {
            const bomResult = result.data;
            const materials = bomResult.bom_lines || [];
            const totalCost = materials.reduce(
              (sum, m) => sum + (m.total_cost_inr || 0),
              0
            );

            return {
              boqItem: boqs[index].boq.material,
              category: boqs[index].boq.category,
              quantity: boqs[index].boq.quantity,
              unit: boqs[index].boq.unit,
              materials,
              totalCost,
              wastageFactor: bomResult.wastage_model_val_mape || 0
            };
          }
          return null;
        }).filter(item => item !== null);

        setBomData([...bomData, ...newBOMItems]);
        setSuccess(`Successfully generated BOM for ${newBOMItems.length} items`);
      } else {
        setError('Failed to generate batch BOM. Please try again.');
      }
    } catch (err) {
      setError(err.message || 'Failed to generate batch BOM');
    } finally {
      setLoading(false);
    }
  };

  // Get material substitutions
  const handleGetSubstitutions = async (materialName) => {
    try {
      const response = await bomAPI.getSubstitutions(materialName);
      if (response.success && response.data) {
        // Handle substitutions display
        console.log('Substitutions:', response.data);
      }
    } catch (err) {
      console.error('Error getting substitutions:', err);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">BOM Generator with Market Intelligence</h1>
          <p className="text-gray-600 mt-1">Generate Bill of Materials from BOQ items with AI-powered recommendations</p>
        </div>
      </div>

      {/* Input Form */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h2 className="font-semibold text-gray-900 mb-4">Generate BOM from BOQ Item</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Category *
            </label>
            <select
              value={boqInput.category}
              onChange={(e) => setBoqInput({ ...boqInput, category: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Select Category</option>
              <option value="Interior">Interior</option>
              <option value="Civil">Civil</option>
              <option value="Electrical">Electrical</option>
              <option value="Plumbing">Plumbing</option>
              <option value="HVAC">HVAC</option>
              <option value="Furniture">Furniture</option>
              <option value="General">General</option>
              <option value="Tiling">Tiling</option>
              <option value="Painting">Painting</option>
              <option value="Carpentry">Carpentry</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              BOQ Item Description *
            </label>
            <input
              type="text"
              value={boqInput.item}
              onChange={(e) => setBoqInput({ ...boqInput, item: e.target.value })}
              placeholder="e.g., 100 sq ft wooden partition"
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
              Unit
            </label>
            <select
              value={boqInput.unit}
              onChange={(e) => setBoqInput({ ...boqInput, unit: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {/* Common BOQ units aligned with dataset (boq_bom_dataset.csv) */}
              <option value="Nos">Nos</option>
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
          </div>
        </div>
        <div className="mt-4">
          <div className="flex gap-3 mb-2">
            <button
              onClick={handleGenerateBOM}
              disabled={loading || !boqInput.category || !boqInput.item}
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
                  Generate BOM (Individual)
                </>
              )}
            </button>
            {boqItemsFromStorage.length > 0 && (
              <button
                onClick={() => handleGenerateBatchBOM(boqItemsFromStorage)}
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
                    Generate BOM for All ({boqItemsFromStorage.length} items)
                  </>
                )}
              </button>
            )}
          </div>
          <p className="text-xs text-gray-500 mt-1">
            <strong>Individual:</strong> Uses the form fields above. <strong>Batch:</strong> Uses each BOQ item's own data (category, quantity, unit, etc.).
          </p>
        </div>
      </div>

      {/* BOQ Items from Storage */}
      {boqItemsFromStorage.length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-gray-900 mb-2">BOQ Items from Generator ({boqItemsFromStorage.length})</h3>
          <p className="text-sm text-gray-700 mb-3">
            Click on any item below to use it in the form above. You can also generate BOM for all items at once using the button above.
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

      {/* BOM Results */}
      {bomData.length > 0 && (
        <div className="space-y-4">
          {bomData.map((bom, index) => (
            <div key={index} className="bg-white rounded-lg border border-gray-200">
              <div className="p-4 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <h3 className="font-semibold text-gray-900">{bom.boqItem}</h3>
                    <div className="flex items-center gap-4 mt-2 text-sm text-gray-600">
                      <span>Category: {bom.category}</span>
                      <span>Quantity: {bom.quantity} {bom.unit}</span>
                      {bom.totalCost > 0 && (
                        <span className="flex items-center gap-1 font-medium text-gray-900">
                          <DollarSign className="w-4 h-4" />
                          Total Cost: ₹{bom.totalCost.toLocaleString('en-IN')}
                        </span>
                      )}
                      {bom.wastageFactor > 0 && (
                        <span className="text-orange-600">
                          Wastage: {bom.wastageFactor}%
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
                <div className="p-4">
                  {/* Materials Table */}
                  {bom.materials && bom.materials.length > 0 ? (
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead className="bg-gray-50 border-b border-gray-200">
                          <tr>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Material</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Quantity</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Unit</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Rate</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Amount</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-600 uppercase">Wastage</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                          {bom.materials.map((material, matIdx) => (
                            <tr key={matIdx} className="hover:bg-gray-50">
                              <td className="px-4 py-3">
                                <div className="font-medium text-gray-900">
                                  {material.material_name || material.name || material.material || 'Unknown'}
                                </div>
                                {material.specification && (
                                  <div className="text-xs text-gray-500 mt-1">{material.specification}</div>
                                )}
                              </td>
                              <td className="px-4 py-3 text-sm text-gray-600">
                                {/* final_quantity is the AI-adjusted quantity including wastage */}
                                {material.final_quantity ?? material.base_quantity ?? 0}
                              </td>
                              <td className="px-4 py-3 text-sm text-gray-600">
                                {material.unit || 'Nos'}
                              </td>
                              <td className="px-4 py-3 text-sm text-gray-600">
                                ₹{material.unit_rate_inr ?? 0}
                              </td>
                              <td className="px-4 py-3 font-medium text-gray-900">
                                ₹{(material.total_cost_inr ?? 0).toLocaleString('en-IN')}
                              </td>
                              <td className="px-4 py-3 text-sm text-gray-600">
                                {material.predicted_wastage_percent ?? 0}%
                              </td>
                            </tr>
                          ))}
                        </tbody>
                        <tfoot className="bg-gray-50">
                          <tr>
                            <td colSpan="4" className="px-4 py-3 font-semibold text-gray-900 text-right">
                              Total:
                            </td>
                            <td className="px-4 py-3 font-semibold text-gray-900">
                              ₹{bom.totalCost.toLocaleString('en-IN')}
                            </td>
                            <td></td>
                          </tr>
                        </tfoot>
                      </table>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      No materials found for this BOQ item
                    </div>
                  )}

                  {/* Market Intelligence Section */}
                  {bom.materials && bom.materials.length > 0 && (
                    <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg">
                      <div className="flex items-start gap-3">
                        <TrendingUp className="w-5 h-5 text-blue-600 mt-0.5" />
                        <div className="flex-1">
                          <h4 className="font-semibold text-gray-900 mb-2">Market Intelligence</h4>
                          <p className="text-sm text-gray-700">
                            Rates are based on current market data and historical purchase patterns. 
                            Consider bulk purchasing for better rates.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {bomData.length === 0 && !loading && (
        <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
          <Package className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">No BOM generated yet. Enter a BOQ item to generate BOM.</p>
        </div>
      )}
    </div>
  );
};

export default BOMGenerator;

