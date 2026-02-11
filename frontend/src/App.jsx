import { useState, useEffect, useCallback } from 'react'
import './App.css'

const API_BASE = '/api'

const TABS = [
  { id: 'predict', label: 'Price Predictor' },
  { id: 'features', label: 'Feature Store' },
]

const CATEGORIES = ['Tops', 'Bottoms', 'Dresses', 'Outerwear', 'Footwear']
const BRANDS = ['Zara', 'H&M', 'Levi\'s', 'Uniqlo', 'Converse', 'North Face', 'Mango', 'Massimo Dutti', 'Steve Madden']
const SUBCATEGORIES = ['T-Shirts', 'Sweaters', 'Coats', 'Jackets', 'Jeans', 'Pants', 'Dresses', 'Boots', 'Sneakers', 'Tops']
const COLORS = ['Black', 'White', 'Navy', 'Gray', 'Blue', 'Brown', 'Pink', 'Cream']
const SEASONS = ['All', 'Spring', 'Summer', 'Fall', 'Winter']
const REGIONS = ['North', 'South', 'East', 'West', 'International']
const AGE_GROUPS = ['18-24', '25-34', '35-44', '45-54', '55+']

function getLast7Days() {
  const days = []
  for (let i = 6; i >= 0; i--) {
    const d = new Date()
    d.setDate(d.getDate() - i)
    days.push(d.toISOString().slice(0, 10))
  }
  return days
}

const defaultProduct = {
  product_id: `P${Date.now().toString(36).slice(-4).toUpperCase()}`,
  name: '',
  category: 'Tops',
  brand: 'Zara',
  subcategory: 'T-Shirts',
  color: 'Black',
  original_price_usd: '',
  season: 'All',
  region: 'North',
  age_group: '25-34',
  inventory_level: 50,
}

const defaultPrices = getLast7Days().map(date => ({ date, price_usd: '' }))

function ModelMetrics({ metrics, onRefresh }) {
  if (!metrics) return null
  const { f1, accuracy, n_samples } = metrics
  if (f1 == null && accuracy == null) return null
  return (
    <div className="model-metrics">
      {f1 != null && <span title="F1 score">F1: {(f1 * 100).toFixed(1)}%</span>}
      {accuracy != null && <span title="Accuracy">Acc: {(accuracy * 100).toFixed(1)}%</span>}
      {n_samples != null && <span>n={n_samples}</span>}
      {onRefresh && (
        <button type="button" className="refresh-btn" onClick={onRefresh} title="Retrain model">
          Retrain
        </button>
      )}
    </div>
  )
}

function FeatureStoreView() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [refreshKey, setRefreshKey] = useState(0)

  const refetch = useCallback(() => {
    setRefreshKey(k => k + 1)
  }, [])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    fetch(`${API_BASE}/features`)
      .then(res => res.ok ? res.json() : Promise.reject(new Error(res.statusText)))
      .then(json => { if (!cancelled) setData(json) })
      .catch(e => { if (!cancelled) setError(e.message) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [refreshKey])

  if (loading) return <main className="main main--single"><p className="status">Loading feature store…</p></main>
  if (error) return <main className="main main--single"><p className="error">{error}</p></main>
  if (!data?.products?.length) return <main className="main main--single"><p className="hint">No features in store. Run generate_feast_data.py and feast materialize.</p></main>

  const featureNames = data.feature_names || []
  return (
    <main className="main main--single">
      <section className="feature-store-section">
        <div className="feature-store-header">
          <div>
            <h2>Feature Store</h2>
            <p className="hint">Features for price prediction model (from Feast)</p>
          </div>
          <button type="button" className="refresh-btn" onClick={refetch}>Refresh</button>
        </div>
        <div className="table-wrap">
          <table className="feature-table">
            <thead>
              <tr>
                <th>Product ID</th>
                {featureNames.map(f => <th key={f}>{f}</th>)}
              </tr>
            </thead>
            <tbody>
              {data.products.map((row, i) => (
                <tr key={row.product_id || i}>
                  <td className="product-id">{row.product_id}</td>
                  {featureNames.map(f => (
                    <td key={f}>{row[f] != null ? (typeof row[f] === 'number' ? row[f].toFixed(2) : row[f]) : '—'}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  )
}

function App() {
  const [activeTab, setActiveTab] = useState('predict')
  const [product, setProduct] = useState(defaultProduct)
  const [prices, setPrices] = useState(defaultPrices)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [addRetrainLoading, setAddRetrainLoading] = useState(false)

  const fetchMetrics = useCallback(() => {
    fetch(`${API_BASE}/model/metrics`)
      .then(res => res.ok ? res.json() : null)
      .then(setMetrics)
      .catch(() => setMetrics(null))
  }, [])

  useEffect(() => { fetchMetrics() }, [fetchMetrics])

  const canPredict = product.name.trim() &&
    product.original_price_usd > 0 &&
    prices.every(p => p.price_usd !== '' && Number(p.price_usd) >= 0)

  const fetchPrediction = useCallback(async () => {
    if (!canPredict) {
      setPrediction(null)
      return
    }
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...product,
          original_price_usd: Number(product.original_price_usd),
          region: product.region || 'North',
          age_group: product.age_group || '25-34',
          price_history: prices.map(p => ({
            date: p.date,
            price_usd: Number(p.price_usd),
          })),
        }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText)
      }
      const data = await res.json()
      setPrediction(data)
    } catch (e) {
      setError(e.message)
      setPrediction(null)
    } finally {
      setLoading(false)
    }
  }, [product, prices, canPredict])

  // Debounced real-time prediction (500ms after last change)
  useEffect(() => {
    const t = setTimeout(fetchPrediction, 500)
    return () => clearTimeout(t)
  }, [fetchPrediction])

  const updateProduct = (field, value) => {
    setProduct(prev => ({ ...prev, [field]: value }))
  }

  const updatePrice = (idx, value) => {
    setPrices(prev => prev.map((p, i) => i === idx ? { ...p, price_usd: value } : p))
  }

  const resetForm = () => {
    setProduct({ ...defaultProduct, product_id: `P${Date.now().toString(36).slice(-4).toUpperCase()}` })
    setPrices(defaultPrices)
    setPrediction(null)
    setError(null)
  }

  const addProductAndRetrain = async () => {
    if (!canPredict) return
    setAddRetrainLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/products`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...product,
          original_price_usd: Number(product.original_price_usd),
          region: product.region || 'North',
          age_group: product.age_group || '25-34',
          inventory_level: product.inventory_level,
          price_history: prices.map(p => ({ date: p.date, price_usd: Number(p.price_usd) })),
        }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText)
      }
      const data = await res.json()
      setMetrics({ f1: data.f1, accuracy: data.accuracy, n_samples: data.n_samples })
      resetForm()
    } catch (e) {
      setError(e.message)
    } finally {
      setAddRetrainLoading(false)
    }
  }

  const retrainModel = async () => {
    setAddRetrainLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/retrain`, { method: 'POST' })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText)
      }
      const data = await res.json()
      setMetrics({ f1: data.f1, accuracy: data.accuracy, n_samples: data.n_samples })
    } catch (e) {
      setError(e.message)
    } finally {
      setAddRetrainLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Fashion Price Predictor</h1>
        <p>Add product data and see price drop predictions in real time</p>
        <ModelMetrics metrics={metrics} onRefresh={retrainModel} />
        <nav className="tabs">
          {TABS.map(({ id, label }) => (
            <button
              key={id}
              className={`tab ${activeTab === id ? 'active' : ''}`}
              onClick={() => setActiveTab(id)}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>

      {activeTab === 'predict' && (
      <main className="main">
        <section className="form-section">
          <h2>Product details</h2>
          <form className="form" onSubmit={e => e.preventDefault()}>
            <div className="row">
              <label>
                Product ID
                <input
                  type="text"
                  value={product.product_id}
                  onChange={e => updateProduct('product_id', e.target.value)}
                />
              </label>
              <label>
                Name
                <input
                  type="text"
                  value={product.name}
                  onChange={e => updateProduct('name', e.target.value)}
                  placeholder="e.g. Classic Wool Coat"
                />
              </label>
            </div>
            <div className="row">
              <label>
                Category
                <select value={product.category} onChange={e => updateProduct('category', e.target.value)}>
                  {CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>
              <label>
                Brand
                <select value={product.brand} onChange={e => updateProduct('brand', e.target.value)}>
                  {BRANDS.map(b => <option key={b} value={b}>{b}</option>)}
                </select>
              </label>
              <label>
                Subcategory
                <select value={product.subcategory} onChange={e => updateProduct('subcategory', e.target.value)}>
                  {SUBCATEGORIES.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </label>
            </div>
            <div className="row">
              <label>
                Color
                <select value={product.color} onChange={e => updateProduct('color', e.target.value)}>
                  {COLORS.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>
              <label>
                Season
                <select value={product.season} onChange={e => updateProduct('season', e.target.value)}>
                  {SEASONS.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </label>
              <label>
                Region
                <select value={product.region} onChange={e => updateProduct('region', e.target.value)}>
                  {REGIONS.map(r => <option key={r} value={r}>{r}</option>)}
                </select>
              </label>
              <label>
                Age group
                <select value={product.age_group} onChange={e => updateProduct('age_group', e.target.value)}>
                  {AGE_GROUPS.map(a => <option key={a} value={a}>{a}</option>)}
                </select>
              </label>
              <label>
                Original price (USD)
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  value={product.original_price_usd}
                  onChange={e => updateProduct('original_price_usd', e.target.value)}
                  placeholder="e.g. 89.99"
                />
              </label>
              <label>
                Inventory level
                <div className="inventory-input">
                  <input
                    type="range"
                    min="1"
                    max="100"
                    value={product.inventory_level}
                    onChange={e => updateProduct('inventory_level', Number(e.target.value))}
                  />
                  <span>{product.inventory_level}</span>
                </div>
              </label>
            </div>
          </form>

          <h3>7-day price history</h3>
          <div className="price-grid">
            {prices.map((p, i) => (
              <label key={p.date}>
                {p.date}
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  value={p.price_usd}
                  onChange={e => updatePrice(i, e.target.value)}
                  placeholder="0.00"
                />
              </label>
            ))}
          </div>
          <div className="form-actions">
            <button className="reset-btn" onClick={resetForm}>Reset form</button>
            <button
              className="add-retrain-btn"
              onClick={addProductAndRetrain}
              disabled={!canPredict || addRetrainLoading}
            >
              {addRetrainLoading ? 'Retraining…' : 'Add product & retrain'}
            </button>
          </div>
        </section>

        <section className="prediction-section">
          <h2>Price prediction</h2>
          {!canPredict && (
            <p className="hint">Fill in all product fields and 7 days of prices to see predictions.</p>
          )}
          {loading && canPredict && <p className="status">Predicting…</p>}
          {error && <p className="error">{error}</p>}
          {prediction && !loading && (
            <div className="prediction-card">
              <div className={`badge ${prediction.predicted_drop ? 'drop' : 'stable'}`}>
                {prediction.predicted_drop ? 'Likely drop' : 'Stable'}
              </div>
              <div className="metric">
                <span className="label">Probability of price drop</span>
                <span className="value">{(prediction.prob_price_drop * 100).toFixed(1)}%</span>
              </div>
              {prediction.price_change_pct_7d != null && (
                <div className="metric">
                  <span className="label">7-day price change</span>
                  <span className="value">{prediction.price_change_pct_7d}%</span>
                </div>
              )}
              <div className="metric">
                <span className="label">Inventory level</span>
                <span className="value">{product.inventory_level}/100</span>
              </div>
              {prediction.recommended_price != null && (
                <div className="metric">
                  <span className="label">Recommended price</span>
                  <span className="value">
                    ${prediction.recommended_price.toFixed(2)}
                    {prediction.recommended_discount_pct != null && (
                      <span className="subvalue"> ({prediction.recommended_discount_pct.toFixed(1)}% off)</span>
                    )}
                  </span>
                </div>
              )}
              <p className="recommendation">{prediction.recommendation}</p>
            </div>
          )}
        </section>
      </main>
      )}

      {activeTab === 'features' && (
        <FeatureStoreView />
      )}

      <footer className="footer">
        <p>Fashion Price Prediction · Powered by scikit-learn</p>
      </footer>
    </div>
  )
}

export default App
