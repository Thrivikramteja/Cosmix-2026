import { useState } from 'react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const toPercent = (value, max = 1) => {
  const safeMax = max > 0 ? max : 1
  const normalized = (Number(value) / safeMax) * 100
  return Math.max(0, Math.min(100, normalized))
}

const toDisplayNumber = (value, decimals = 2) => {
  const number = Number(value)
  if (!Number.isFinite(number)) return '0'
  return number.toFixed(decimals)
}

const metricLabels = {
  mae: 'MAE (m)',
  rmse: 'RMSE (m)',
  r2_score: 'R2 Score',
  delta1: 'Delta1',
  delta2: 'Delta2',
  iou: 'IoU',
  f1_score: 'F1 Score',
  precision: 'Precision',
  recall: 'Recall',
  buildings_detected: 'Buildings Detected',
  within25_ratio: 'Within 25% Threshold (Count/Detected)',
}

const higherIsBetter = new Set([
  'r2_score',
  'delta1',
  'delta2',
  'iou',
  'f1_score',
  'precision',
  'recall',
  'buildings_detected',
  'within25_ratio',
])

const lowerIsBetter = new Set(['mae', 'rmse'])

const calculateDeltaMetrics = (buildings) => {
  if (!Array.isArray(buildings) || buildings.length === 0) {
    return { delta1: 0, delta2: 0 }
  }

  let countDelta1 = 0
  let countDelta2 = 0

  buildings.forEach((building) => {
    const pred = Number(building.predicted_height_m)
    const actual = Number(building.actual_height_m)

    if (!Number.isFinite(pred) || !Number.isFinite(actual) || pred <= 0 || actual <= 0) {
      return
    }

    const ratio = Math.max(pred / actual, actual / pred)
    if (ratio < 1.25) countDelta1 += 1
    if (ratio < 1.25 * 1.25) countDelta2 += 1
  })

  const total = buildings.length || 1
  return {
    delta1: Number((countDelta1 / total).toFixed(2)),
    delta2: Number((countDelta2 / total).toFixed(2)),
  }
}

const calculateWithin25Stats = (buildings) => {
  if (!Array.isArray(buildings) || buildings.length === 0) {
    return { count: 0, total: 0, ratio: 0 }
  }

  let count = 0

  buildings.forEach((building) => {
    const pred = Number(building.predicted_height_m)
    const actual = Number(building.actual_height_m)

    if (!Number.isFinite(pred) || !Number.isFinite(actual) || pred <= 0 || actual <= 0) {
      return
    }

    const ratio = Math.max(pred / actual, actual / pred)
    if (ratio < 1.25) count += 1
  })

  const total = buildings.length
  return {
    count,
    total,
    ratio: Number((count / (total || 1)).toFixed(2)),
  }
}

function App() {
  const [imageId, setImageId] = useState('')
  const [opticalFile, setOpticalFile] = useState(null)
  const [sarFile, setSarFile] = useState(null)
  const [csvFile, setCsvFile] = useState(null)
  const [selectedModel, setSelectedModel] = useState('advanced')
  const [singlePrediction, setSinglePrediction] = useState(null)
  const [comparePrediction, setComparePrediction] = useState(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState('')

  const clearResultsAndErrors = () => {
    setSinglePrediction(null)
    setComparePrediction(null)
    setError('')
  }

  const handleOpticalFileChange = (event) => {
    const file = event.target.files?.[0] || null
    setOpticalFile(file)
    clearResultsAndErrors()
  }

  const handleSarFileChange = (event) => {
    const file = event.target.files?.[0] || null
    setSarFile(file)
    clearResultsAndErrors()
  }

  const handleCsvFileChange = (event) => {
    const file = event.target.files?.[0] || null
    setCsvFile(file)
    clearResultsAndErrors()
  }

  const buildFormData = () => {
    const formData = new FormData()
    formData.append('image_id', imageId.trim())
    formData.append('optical_file', opticalFile)
    formData.append('sar_file', sarFile)
    formData.append('csv_file', csvFile)
    return formData
  }

  const requestPrediction = async (route) => {
    const response = await fetch(`${API_BASE_URL}${route}`, {
      method: 'POST',
      body: buildFormData(),
    })

    if (!response.ok) {
      const message = await response.text()
      throw new Error(message || `Request failed with status ${response.status}`)
    }

    return response.json()
  }

  const collectMetrics = (prediction) => {
    const summary = prediction?.data_summary ?? {}
    const regression = prediction?.metrics?.regression_heights ?? {}
    const segmentation = prediction?.metrics?.segmentation_footprints ?? {}
    const buildings = Array.isArray(prediction?.buildings_data) ? prediction.buildings_data : []
    const deltaFromData = calculateDeltaMetrics(buildings)
    const within25Stats = calculateWithin25Stats(buildings)

    return {
      mae: Number(regression.mae ?? 0),
      rmse: Number(regression.rmse ?? 0),
      r2_score: Number(regression.r2_score ?? 0),
      delta1: Number(regression.delta1 ?? deltaFromData.delta1 ?? 0),
      delta2: Number(regression.delta2 ?? deltaFromData.delta2 ?? 0),
      iou: Number(segmentation.iou ?? 0),
      f1_score: Number(segmentation.f1_score ?? 0),
      precision: Number(segmentation.precision ?? 0),
      recall: Number(segmentation.recall ?? 0),
      buildings_detected: Number(summary.buildings_detected ?? 0),
      within25_ratio: Number(within25Stats.ratio ?? 0),
      within25_count: Number(within25Stats.count ?? 0),
      within25_total: Number(within25Stats.total ?? 0),
    }
  }

  const getWinner = (key, basicValue, advancedValue) => {
    if (basicValue === advancedValue) return 'Tie'
    if (higherIsBetter.has(key)) {
      return basicValue > advancedValue ? 'Baseline' : 'FusionHeightNet'
    }
    if (lowerIsBetter.has(key)) {
      return basicValue < advancedValue ? 'Baseline' : 'FusionHeightNet'
    }
    return 'Tie'
  }

  const buildComparisonRows = (basicPrediction, advancedPrediction) => {
    const basic = collectMetrics(basicPrediction)
    const advanced = collectMetrics(advancedPrediction)

    return Object.keys(metricLabels).map((key) => ({
      key,
      label: metricLabels[key],
      basic: basic[key],
      advanced: advanced[key],
      winner: getWinner(key, basic[key], advanced[key]),
      basicWithin25Count: basic.within25_count,
      basicWithin25Total: basic.within25_total,
      advancedWithin25Count: advanced.within25_count,
      advancedWithin25Total: advanced.within25_total,
    }))
  }

  const handleImageUpload = async (event) => {
    event.preventDefault()
    if (!imageId.trim() || !opticalFile || !sarFile || !csvFile) return

    try {
      setIsSubmitting(true)
      setError('')

      if (selectedModel === 'compare') {
        const [basicData, advancedData] = await Promise.all([
          requestPrediction('/predict/basic'),
          requestPrediction('/predict/advanced'),
        ])

        setSinglePrediction(null)
        setComparePrediction({
          basic: basicData,
          advanced: advancedData,
          rows: buildComparisonRows(basicData, advancedData),
        })
      } else {
        const route = selectedModel === 'advanced' ? '/predict/advanced' : '/predict/basic'
        const data = await requestPrediction(route)
        setComparePrediction(null)
        setSinglePrediction(data)
      }
    } catch (uploadError) {
      console.error('Error connecting to backend:', uploadError)
      setError(
        uploadError?.message ||
          'Unable to connect to backend. Make sure FastAPI is running on port 8000.',
      )
    } finally {
      setIsSubmitting(false)
    }
  }

  const renderSinglePrediction = (title, prediction) => {
    const summary = prediction?.data_summary ?? {}
    const regression = prediction?.metrics?.regression_heights ?? {}
    const segmentation = prediction?.metrics?.segmentation_footprints ?? {}
    const buildings = Array.isArray(prediction?.buildings_data) ? prediction.buildings_data : []
    const visualizations = prediction?.visualizations ?? {}
    const deltaFromData = calculateDeltaMetrics(buildings)
    const within25Stats = calculateWithin25Stats(buildings)
    const regressionMetrics = {
      mae: regression.mae ?? 0,
      rmse: regression.rmse ?? 0,
      r2_score: regression.r2_score ?? 0,
      delta1: regression.delta1 ?? deltaFromData.delta1,
      delta2: regression.delta2 ?? deltaFromData.delta2,
    }

    const maxDisplayedHeight = buildings.reduce(
      (max, b) => Math.max(max, Number(b.predicted_height_m) || 0, Number(b.actual_height_m) || 0),
      1,
    )

    const segmentationBars = Object.entries(segmentation).map(([key, value]) => ({
      label: key.replaceAll('_', ' ').toUpperCase(),
      value: Number(value) || 0,
    }))

    return (
      <div className="result">
        <h2>{title}</h2>

        <div className="stats-grid">
          <div className="stat-card">
            <p className="stat-label">Status</p>
            <p className="stat-value">{prediction.status || 'unknown'}</p>
          </div>
          <div className="stat-card">
            <p className="stat-label">Buildings Detected</p>
            <p className="stat-value">{summary.buildings_detected ?? 0}</p>
          </div>
          <div className="stat-card">
            <p className="stat-label">MAE (m)</p>
            <p className="stat-value">{toDisplayNumber(regressionMetrics.mae)}</p>
          </div>
          <div className="stat-card">
            <p className="stat-label">RMSE (m)</p>
            <p className="stat-value">{toDisplayNumber(regressionMetrics.rmse)}</p>
          </div>
          <div className="stat-card">
            <p className="stat-label">R2 Score</p>
            <p className="stat-value">{toDisplayNumber(regressionMetrics.r2_score)}</p>
          </div>
          <div className="stat-card">
            <p className="stat-label">Delta1</p>
            <p className="stat-value">{toDisplayNumber(regressionMetrics.delta1)}</p>
          </div>
          <div className="stat-card">
            <p className="stat-label">Delta2</p>
            <p className="stat-value">{toDisplayNumber(regressionMetrics.delta2)}</p>
          </div>
          <div className="stat-card">
            <p className="stat-label">Within 25% of Actual Height</p>
            <p className="stat-value">
              {within25Stats.count}/{within25Stats.total}
            </p>
          </div>
        </div>

        <div className="charts-grid">
          <div className="chart-card">
            <h3>Footprint Extraction Metrics</h3>
            <div className="metric-bars">
              {segmentationBars.length > 0 ? (
                segmentationBars.map((item) => (
                  <div className="metric-row" key={item.label}>
                    <div className="metric-head">
                      <span>{item.label}</span>
                      <span>{toDisplayNumber(item.value)}</span>
                    </div>
                    <div className="metric-track">
                      <div className="metric-fill" style={{ width: `${toPercent(item.value)}%` }} />
                    </div>
                  </div>
                ))
              ) : (
                <div className="metric-row">
                  <p className="empty-state">No segmentation metrics available.</p>
                </div>
              )}
            </div>
          </div>

          <div className="chart-card">
            <h3>Height Estimation vs Reference (Top 10)</h3>
            <div className="height-bars">
              {buildings.length > 0 ? (
                buildings.slice(0, 10).map((building) => (
                  <div className="height-row" key={building.rank}>
                    <div className="height-title">Building {building.rank}</div>
                    <div className="height-track-group">
                      <div className="height-track">
                        <div
                          className="height-fill predicted"
                          style={{
                            width: `${toPercent(building.predicted_height_m, maxDisplayedHeight)}%`,
                          }}
                        />
                      </div>
                      <span className="height-value">Pred: {building.predicted_height_m}m</span>
                    </div>
                    <div className="height-track-group">
                      <div className="height-track">
                        <div
                          className="height-fill actual"
                          style={{
                            width: `${toPercent(building.actual_height_m, maxDisplayedHeight)}%`,
                          }}
                        />
                      </div>
                      <span className="height-value">Actual: {building.actual_height_m}m</span>
                    </div>
                  </div>
                ))
              ) : (
                <p className="empty-state">No building rows available for comparison.</p>
              )}
            </div>
          </div>
        </div>

        {buildings.length > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Predicted Height (m)</th>
                  <th>Actual Height (m)</th>
                </tr>
              </thead>
              <tbody>
                {buildings.slice(0, 10).map((building) => (
                  <tr key={building.rank}>
                    <td>{building.rank}</td>
                    <td>{building.predicted_height_m}</td>
                    <td>{building.actual_height_m}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="images-grid">
          {visualizations.image_contours_b64 && (
            <figure className="result-image-card">
              <img src={visualizations.image_contours_b64} alt="Predicted building contours" />
              <figcaption>EO Image with Predicted Footprint Contours</figcaption>
            </figure>
          )}

          {visualizations.image_heatmap_b64 && (
            <figure className="result-image-card">
              <img src={visualizations.image_heatmap_b64} alt="Predicted height heatmap" />
              <figcaption>Semantic-Refined Height Estimation Heatmap</figcaption>
            </figure>
          )}
        </div>
      </div>
    )
  }

  return (
    <main className="page">
      <section className="card">
        <p className="eyebrow">FusionHeightNet: Multi-Level Cross-Fusion</p>
        <h1>FusionHeightNet</h1>
        <p className="subtitle">
          Multi-source remote sensing inference with multi-task joint learning of footprint extraction and
          height estimation, plus semantic refinement on SpaceNet-6.
        </p>

        <form className="upload-form" onSubmit={handleImageUpload}>
          <label className="model-input" htmlFor="modelSelect">
            <span>Model Route</span>
            <select
              id="modelSelect"
              value={selectedModel}
              onChange={(event) => {
                setSelectedModel(event.target.value)
                clearResultsAndErrors()
              }}
            >
              <option value="basic">Basic Early Fusion (/predict/basic)</option>
              <option value="advanced">Advanced Cross-Attention (/predict/advanced)</option>
              <option value="compare">Compare Both Models (calls both endpoints)</option>
            </select>
          </label>

          <label className="model-input" htmlFor="imageIdInput">
            <span>Image ID (must match CSV ImageId)</span>
            <input
              id="imageIdInput"
              type="text"
              value={imageId}
              onChange={(event) => {
                setImageId(event.target.value)
                clearResultsAndErrors()
              }}
              placeholder="e.g. 20190804111224_20190804111453_tile_8681"
            />
          </label>

          <label className="file-input" htmlFor="opticalUpload">
            <span>
              Optical image: {opticalFile ? opticalFile.name : 'choose image file'}
            </span>
            <input
              id="opticalUpload"
              type="file"
              accept=".tif,.tiff,image/*"
              onChange={handleOpticalFileChange}
            />
          </label>

          <label className="file-input" htmlFor="sarUpload">
            <span>SAR image: {sarFile ? sarFile.name : 'choose image file'}</span>
            <input
              id="sarUpload"
              type="file"
              accept=".tif,.tiff,image/*"
              onChange={handleSarFileChange}
            />
          </label>

          <label className="file-input" htmlFor="csvUpload">
            <span>Ground Truth CSV: {csvFile ? csvFile.name : 'choose csv file'}</span>
            <input id="csvUpload" type="file" accept=".csv,text/csv" onChange={handleCsvFileChange} />
          </label>

          <button
            type="submit"
            disabled={!imageId.trim() || !opticalFile || !sarFile || !csvFile || isSubmitting}
          >
            {isSubmitting
              ? 'Submitting...'
              : selectedModel === 'compare'
                ? 'Run Both Models'
                : 'Submit'}
          </button>
        </form>

        {error && <p className="message error">{error}</p>}

        {singlePrediction &&
          renderSinglePrediction(
            selectedModel === 'advanced'
              ? 'FusionHeightNet Inference Output'
              : 'Early-Fusion Baseline Inference Output',
            singlePrediction,
          )}

        {comparePrediction && (
          <div className="result">
            <h2>FusionHeightNet vs Early-Fusion Baseline (SpaceNet-6 Metrics)</h2>

            <div className="table-wrap comparison-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Early-Fusion Baseline</th>
                    <th>FusionHeightNet</th>
                    <th>Winner</th>
                  </tr>
                </thead>
                <tbody>
                  {comparePrediction.rows.map((row) => (
                    <tr key={row.key}>
                      <td>{row.label}</td>
                      <td>
                        {row.key === 'within25_ratio'
                          ? `${row.basicWithin25Count}/${row.basicWithin25Total} (${toDisplayNumber(row.basic)})`
                          : toDisplayNumber(row.basic)}
                      </td>
                      <td>
                        {row.key === 'within25_ratio'
                          ? `${row.advancedWithin25Count}/${row.advancedWithin25Total} (${toDisplayNumber(row.advanced)})`
                          : toDisplayNumber(row.advanced)}
                      </td>
                      <td>
                        <span
                          className={
                            row.winner === 'Tie'
                              ? 'winner-chip tie'
                              : row.winner === 'Baseline'
                                ? 'winner-chip basic'
                                : 'winner-chip advanced'
                          }
                        >
                          {row.winner}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="compare-panels">
              {renderSinglePrediction(
                'Early-Fusion Baseline (/predict/basic)',
                comparePrediction.basic,
              )}
              {renderSinglePrediction(
                'FusionHeightNet (/predict/advanced)',
                comparePrediction.advanced,
              )}
            </div>
          </div>
        )}
      </section>
    </main>
  )
}

export default App
