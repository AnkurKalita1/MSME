import { useState, useEffect, useRef } from 'react';
import { Plus, Package, Loader2, AlertCircle, CheckCircle, X, ChevronDown, ChevronRight, DollarSign, TrendingUp, RotateCcw } from 'lucide-react';
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
    unit: 'NOS',
    region: 'central',
    season: 'normal',
    grade: 'standard'
  });
  const [expandedItems, setExpandedItems] = useState({});
  const [boqItemsFromStorage, setBoqItemsFromStorage] = useState([]);
  const [selectedBoqIndex, setSelectedBoqIndex] = useState(null);
  
  // Material suggestions state
  const [materialSuggestions, setMaterialSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [availableUnits, setAvailableUnits] = useState(['NOS', 'Sqft', 'Sqm', 'Cft', 'Rft', 'Mtr', 'Kg', 'Ltr', 'Ton', 'Pcs', 'Point']);
  const suggestionsRef = useRef(null);
  
  // Store category per BOQ item
  const [boqItemCategories, setBoqItemCategories] = useState({});
  const [boqItemUnits, setBoqItemUnits] = useState({});

  // Load BOQ items and their saved categories/units from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('boqItems');
    const savedCategories = localStorage.getItem('boqItemCategories');
    const savedUnits = localStorage.getItem('boqItemUnits');
    
    if (saved) {
      try {
        const items = JSON.parse(saved);
        setBoqItemsFromStorage(items);
        
        if (savedCategories) {
          setBoqItemCategories(JSON.parse(savedCategories));
        }
        if (savedUnits) {
          setBoqItemUnits(JSON.parse(savedUnits));
        }
        
        // Pre-fill first item if available
        if (items.length > 0) {
          const firstItem = items[0];
          const savedCategory = savedCategories ? JSON.parse(savedCategories)[0] : firstItem.category || '';
          const savedUnit = savedUnits ? JSON.parse(savedUnits)[0] : firstItem.unit || 'NOS';
          
          setBoqInput({
            category: savedCategory,
            item: firstItem.material || '',
            quantity: firstItem.qty || 1,
            unit: savedUnit,
            region: firstItem.region || 'central',
            season: firstItem.season || 'normal',
            grade: firstItem.grade || 'standard'
          });
          setSelectedBoqIndex(0);
          
          // Load material-specific units
          updateUnitsForMaterial(firstItem.material || '', savedCategory);
        }
      } catch (e) {
        console.error('Failed to load BOQ items:', e);
      }
    }
  }, []);

  // Save categories and units to localStorage whenever they change
  useEffect(() => {
    if (Object.keys(boqItemCategories).length > 0) {
      localStorage.setItem('boqItemCategories', JSON.stringify(boqItemCategories));
    }
    if (Object.keys(boqItemUnits).length > 0) {
      localStorage.setItem('boqItemUnits', JSON.stringify(boqItemUnits));
    }
  }, [boqItemCategories, boqItemUnits]);

  // Update units based on material and category
  const updateUnitsForMaterial = async (material, category) => {
    if (!material && !category) {
      setAvailableUnits(['NOS', 'Sqft', 'Sqm', 'Cft', 'Rft', 'Mtr', 'Kg', 'Ltr', 'Ton', 'Pcs', 'Point']);
      return;
    }

    // Material-specific unit mappings
    const materialLower = material.toLowerCase();
    const categoryLower = (category || '').toLowerCase();
    
    let units = [];
    
    // Check material keywords
    if (materialLower.includes('wire') || materialLower.includes('cable')) {
      units = ['Mtr', 'Rft', 'Kg', 'NOS'];
    } else if (materialLower.includes('conduit') || materialLower.includes('pipe')) {
      units = ['Mtr', 'Rft', 'NOS'];
    } else if (materialLower.includes('tile')) {
      units = ['Sqft', 'Sqm', 'Pcs', 'NOS'];
    } else if (materialLower.includes('paint') || materialLower.includes('primer')) {
      units = ['Ltr', 'Kg', 'Sqft'];
    } else if (materialLower.includes('putty') || materialLower.includes('adhesive') || materialLower.includes('grout')) {
      units = ['Kg', 'Ltr', 'Sqft'];
    } else if (materialLower.includes('plywood') || materialLower.includes('laminate')) {
      units = ['Sqft', 'Sqm', 'Pcs', 'Mtr'];
    } else if (materialLower.includes('cement')) {
      units = ['Kg', 'Ton', 'Bag', 'NOS'];
    } else if (materialLower.includes('sand')) {
      units = ['Cft', 'Ton', 'Kg'];
    } else if (materialLower.includes('brick')) {
      units = ['NOS', 'Pcs', 'Sqft'];
    } else if (materialLower.includes('steel') || materialLower.includes('rebar')) {
      units = ['Kg', 'Ton', 'Mtr'];
    } else {
      // Category-based defaults
      if (categoryLower === 'electrical') {
        units = ['Mtr', 'Rft', 'NOS', 'Pcs', 'Kg'];
      } else if (categoryLower === 'plumbing') {
        units = ['Mtr', 'Rft', 'NOS', 'Pcs', 'Kg', 'Ltr'];
      } else if (categoryLower === 'tiling') {
        units = ['Sqft', 'Sqm', 'Pcs', 'Kg'];
      } else if (categoryLower === 'painting') {
        units = ['Ltr', 'Kg', 'Sqft'];
      } else if (categoryLower === 'carpentry') {
        units = ['Sqft', 'Sqm', 'Pcs', 'Mtr', 'Rft'];
      } else {
        units = ['NOS', 'Sqft', 'Sqm', 'Mtr', 'Kg', 'Ltr', 'Pcs'];
      }
    }
    
    setAvailableUnits(units);
    
    // Update unit if current unit is not in available units
    if (boqInput.unit && !units.includes(boqInput.unit)) {
      setBoqInput({ ...boqInput, unit: units[0] || 'NOS' });
    }
  };

  // Fetch material suggestions
  const fetchMaterialSuggestions = async (category, searchText) => {
    if (!searchText || searchText.length < 2) {
      setMaterialSuggestions([]);
      setShowSuggestions(false);
      return;
    }

    try {
      const response = await bomAPI.getMaterialSuggestions(category, searchText);
      if (response.success && response.data && response.data.suggestions) {
        setMaterialSuggestions(response.data.suggestions);
        setShowSuggestions(true);
      }
    } catch (err) {
      console.error('Error fetching material suggestions:', err);
    }
  };

  // Handle selecting a BOQ item from the list
  const handleSelectBoqItem = (item, index) => {
    const savedCategory = boqItemCategories[index] || item.category || '';
    const savedUnit = boqItemUnits[index] || item.unit || 'NOS';
    
    setBoqInput({
      category: savedCategory,
      item: item.material || '',
      quantity: item.qty || 1,
      unit: savedUnit,
      region: item.region || 'central',
      season: item.season || 'normal',
      grade: item.grade || 'standard'
    });
    setSelectedBoqIndex(index);
    
    // Update units for this material
    updateUnitsForMaterial(item.material || '', savedCategory);
  };

  // Restore BOQ default values
  const handleRestoreDefaults = () => {
    if (selectedBoqIndex !== null && boqItemsFromStorage[selectedBoqIndex]) {
      const item = boqItemsFromStorage[selectedBoqIndex];
      setBoqInput({
        category: item.category || '',
        item: item.material || '',
        quantity: item.qty || 1,
        unit: item.unit || 'NOS',
        region: item.region || 'central',
        season: item.season || 'normal',
        grade: item.grade || 'standard'
      });
      updateUnitsForMaterial(item.material || '', item.category || '');
    }
  };

  // Handle category change - save per item
  const handleCategoryChange = (category) => {
    setBoqInput({ ...boqInput, category });
    
    // Save category for current item
    if (selectedBoqIndex !== null) {
      setBoqItemCategories({
        ...boqItemCategories,
        [selectedBoqIndex]: category
      });
    }
    
    // Update units based on category
    updateUnitsForMaterial(boqInput.item, category);
  };

  // Handle item description change with autocomplete
  const handleItemChange = (value) => {
    setBoqInput({ ...boqInput, item: value });
    
    // Fetch suggestions
    fetchMaterialSuggestions(boqInput.category, value);
    
    // Update units based on material
    updateUnitsForMaterial(value, boqInput.category);
  };

  // Handle unit change - save per item
  const handleUnitChange = (unit) => {
    setBoqInput({ ...boqInput, unit });
    
    // Save unit for current item
    if (selectedBoqIndex !== null) {
      setBoqItemUnits({
        ...boqItemUnits,
        [selectedBoqIndex]: unit
      });
    }
  };

  // Handle suggestion selection
  const handleSuggestionSelect = (suggestion) => {
    setBoqInput({
      ...boqInput,
      item: suggestion.material_name,
      unit: suggestion.unit || boqInput.unit
    });
    setShowSuggestions(false);
    setMaterialSuggestions([]);
    
    // Update units
    updateUnitsForMaterial(suggestion.material_name, boqInput.category);
  };

  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (suggestionsRef.current && !suggestionsRef.current.contains(event.target)) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

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
        wbs: {
          execution_tasks: []
        }
      };

      const response = await bomAPI.generate(payload);

      if (response.success && response.data) {
        const bomResult = response.data;
        const materials = bomResult.bom_lines || [];
        const totalCost = materials.reduce(
          (sum, m) => sum + (m.total_cost_inr || 0),
          0
        );

        // Create unique BOM with ID
        const newBOM = {
          id: Date.now() + Math.random(), // Unique ID
          boqItem: boqInput.item,
          category: boqInput.category,
          quantity: boqInput.quantity,
          unit: boqInput.unit,
          materials,
          totalCost,
          wastageFactor: bomResult.wastage_model_val_mape || 0
        };

        setBomData([...bomData, newBOM]);
        setSuccess('BOM generated successfully');
        
        // Save category and unit for current item
        if (selectedBoqIndex !== null) {
          setBoqItemCategories({
            ...boqItemCategories,
            [selectedBoqIndex]: boqInput.category
          });
          setBoqItemUnits({
            ...boqItemUnits,
            [selectedBoqIndex]: boqInput.unit
          });
        }
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
      const boqs = boqItems.map((item, index) => ({
        boq: {
          category: boqItemCategories[index] || item.category || 'General',
          material: item.material || item.item || '',
          quantity: item.qty || item.quantity || 1,
          unit: boqItemUnits[index] || item.unit || 'NOS',
          region: item.region || 'central',
          season: item.season || 'normal',
          grade: item.grade || 'standard'
        },
        wbs: {
          execution_tasks: []
        }
      }));

      const response = await bomAPI.generateBatch(boqs);

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
              id: Date.now() + Math.random() + index, // Unique ID
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
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-semibold text-gray-900">Generate BOM from BOQ Item</h2>
          {selectedBoqIndex !== null && (
            <button
              onClick={handleRestoreDefaults}
              className="flex items-center gap-2 px-3 py-1.5 text-sm text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded"
              title="Restore BOQ default values"
            >
              <RotateCcw className="w-4 h-4" />
              Restore Defaults
            </button>
          )}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Category *
            </label>
            <select
              value={boqInput.category}
              onChange={(e) => handleCategoryChange(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Select Category</option>
              <option value="Electrical">Electrical</option>
              <option value="Plumbing">Plumbing</option>
              <option value="Tiling">Tiling</option>
              <option value="Painting">Painting</option>
              <option value="Carpentry">Carpentry</option>
              <option value="Civil">Civil</option>
              <option value="Interior">Interior</option>
              <option value="HVAC">HVAC</option>
              <option value="Furniture">Furniture</option>
              <option value="General">General</option>
            </select>
          </div>
          <div className="relative" ref={suggestionsRef}>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              BOQ Item Description *
            </label>
            <input
              type="text"
              value={boqInput.item}
              onChange={(e) => handleItemChange(e.target.value)}
              onFocus={() => {
                if (materialSuggestions.length > 0) setShowSuggestions(true);
              }}
              placeholder="e.g., Electrical wiring for 3 rooms"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            {showSuggestions && materialSuggestions.length > 0 && (
              <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                {materialSuggestions.map((suggestion, idx) => (
                  <div
                    key={idx}
                    onClick={() => handleSuggestionSelect(suggestion)}
                    className="px-4 py-2 hover:bg-blue-50 cursor-pointer border-b border-gray-100 last:border-b-0"
                  >
                    <div className="font-medium text-gray-900">{suggestion.material_name}</div>
                    {suggestion.unit && (
                      <div className="text-xs text-gray-500">Unit: {suggestion.unit}</div>
                    )}
                  </div>
                ))}
              </div>
            )}
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
              onChange={(e) => handleUnitChange(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {availableUnits.map((unit) => (
                <option key={unit} value={unit}>{unit}</option>
              ))}
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

      {/* Success/Error Messages */}
      {success && (
        <div className="bg-green-50 border border-green-200 text-green-800 px-4 py-3 rounded-lg flex items-center justify-between">
          <div className="flex items-center">
            <CheckCircle className="w-5 h-5 mr-2" />
            {success}
          </div>
          <button onClick={() => setSuccess(null)}>
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg flex items-center justify-between">
          <div className="flex items-center">
            <AlertCircle className="w-5 h-5 mr-2" />
            {error}
          </div>
          <button onClick={() => setError(null)}>
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* BOQ Items from Storage */}
      {boqItemsFromStorage.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-900 mb-2">
            BOQ Items from Generator ({boqItemsFromStorage.length})
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            Click on any item below to use it in the form above. You can also generate BOM for all items at once using the button above.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {boqItemsFromStorage.map((item, idx) => (
              <div
                key={idx}
                onClick={() => handleSelectBoqItem(item, idx)}
                className={`p-3 border-2 rounded-lg cursor-pointer transition-colors ${
                  selectedBoqIndex === idx
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-blue-100 hover:border-blue-300'
                }`}
              >
                <div className="font-medium text-gray-900">{item.material}</div>
                <div className="text-xs text-gray-600 mt-1">
                  Qty: {item.qty} {item.unit || 'NOS'} | Rate: ₹{item.rate_most_likely || 0}
                </div>
                {boqItemCategories[idx] && (
                  <div className="text-xs text-blue-600 mt-1">
                    Category: {boqItemCategories[idx]}
                  </div>
                )}
                {selectedBoqIndex === idx && (
                  <div className="text-xs text-blue-600 font-medium mt-1">✓ Selected</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Generated BOM Items */}
      {bomData.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-xl font-bold text-gray-900">Generated BOM Items</h2>
          {bomData.map((bom, index) => (
            <div key={bom.id || index} className="bg-white rounded-lg border border-gray-200">
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
                          Wastage: {bom.wastageFactor.toFixed(2)}%
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
                                {material.final_quantity ?? material.base_quantity ?? 0}
                              </td>
                              <td className="px-4 py-3 text-sm text-gray-600">
                                {material.unit || 'NOS'}
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
                    <p className="text-gray-500 text-center py-4">No materials found for this BOM.</p>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default BOMGenerator;
