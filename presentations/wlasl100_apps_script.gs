/*
 * WLASL100 results-only Google Slides generator.
 *
 * Setup:
 * 1. Upload wlasl100_slides_data.json to Google Drive.
 * 2. Put its Drive file ID in DATA_FILE_ID.
 * 3. Open script.new, paste this file, and run createWLASL100Deck().
 */

const DATA_FILE_ID = 'PASTE_GOOGLE_DRIVE_JSON_FILE_ID_HERE';

const COLORS = {
  ink: '#102A43',
  navy: '#0B1F33',
  paper: '#F7F4EE',
  white: '#FFFFFF',
  muted: '#5D6D7E',
  gold: '#D09A36',
};

const METRICS = [
  ['Accuracy', 'accuracy'],
  ['Parameter count', 'param_count_tensors'],
  ['Latency', 'latency_ms_per_batch'],
  ['Model size', 'model_size_mb'],
  ['FLOPs', 'flops_per_batch'],
];

function createWLASL100Deck() {
  if (DATA_FILE_ID === 'PASTE_GOOGLE_DRIVE_JSON_FILE_ID_HERE') {
    throw new Error('Set DATA_FILE_ID to the uploaded JSON file ID first.');
  }

  const data = JSON.parse(
    DriveApp.getFileById(DATA_FILE_ID).getBlob().getDataAsString('UTF-8')
  );
  validateData_(data);

  const deck = SlidesApp.create(data.deck_title);
  const firstSlide = deck.getSlides()[0];
  clearSlide_(firstSlide);

  addCoverSlide_(firstSlide, data);
  addBaselineSlide_(newBlankSlide_(deck), data);
  addGhostFp32Slide_(newBlankSlide_(deck), data);
  addGhostQuantizedComparisonSlide_(newBlankSlide_(deck), data);
  addKDAbsoluteSlide_(newBlankSlide_(deck), data);
  addKDFp32ComparisonSlide_(newBlankSlide_(deck), data);
  addKDQuantizedComparisonSlide_(newBlankSlide_(deck), data);
  addLowRankAbsoluteSlide_(newBlankSlide_(deck), data, 'transformer');
  addLowRankComparisonSlide_(newBlankSlide_(deck), data, 'transformer', 'fp32');
  addLowRankComparisonSlide_(newBlankSlide_(deck), data, 'transformer', 'quantized');
  addLowRankAbsoluteSlide_(newBlankSlide_(deck), data, 'all');
  addLowRankComparisonSlide_(newBlankSlide_(deck), data, 'all', 'fp32');
  addLowRankComparisonSlide_(newBlankSlide_(deck), data, 'all', 'quantized');
  addGhostSummarySlide_(newBlankSlide_(deck), data);
  addKDSummarySlide_(newBlankSlide_(deck), data);
  addLowRankSummarySlide_(newBlankSlide_(deck), data);

  Logger.log('Created 16 slides: ' + deck.getUrl());
  return deck.getUrl();
}

function validateData_(data) {
  if (!data || !data.deck_title || !data.baseline || !Array.isArray(data.experiments)) {
    throw new Error('The JSON does not have the expected WLASL100 deck schema.');
  }
}

function newBlankSlide_(deck) {
  return deck.appendSlide(SlidesApp.PredefinedLayout.BLANK);
}

function clearSlide_(slide) {
  slide.getPageElements().forEach(function(element) {
    element.remove();
  });
}

function setSlideBackground_(slide, color) {
  slide.getBackground().setSolidFill(color);
}

function addText_(slide, text, x, y, width, height, options) {
  const shape = slide.insertTextBox(text, x, y, width, height);
  const range = shape.getText();
  const style = range.getTextStyle();
  const paragraph = range.getParagraphStyle();
  const settings = options || {};

  style.setFontFamily(settings.fontFamily || 'Aptos');
  style.setFontSize(settings.fontSize || 18);
  style.setForegroundColor(settings.color || COLORS.ink);
  style.setBold(Boolean(settings.bold));
  paragraph.setParagraphAlignment(
    settings.align || SlidesApp.ParagraphAlignment.START
  );
  shape.getFill().setTransparent();
  return shape;
}

function addRule_(slide, x, y, width, color) {
  const line = slide.insertLine(
    SlidesApp.LineCategory.STRAIGHT, x, y, x + width, y
  );
  line.getLineFill().setSolidFill(color || COLORS.gold);
  line.setWeight(1.4);
}

function addFooter_(slide, text) {
  addText_(slide, text, 40, 377, 640, 13, {
    fontSize: 8.5,
    color: COLORS.muted,
    align: SlidesApp.ParagraphAlignment.CENTER,
  });
}

function addSlideHeading_(slide, title, subtitle) {
  addText_(slide, title, 40, 26, 640, 33, {
    fontFamily: 'Aptos Display',
    fontSize: 27,
    color: COLORS.ink,
    bold: true,
  });
  addText_(slide, subtitle, 40, 66, 640, 20, {
    fontSize: 12.5,
    color: COLORS.muted,
  });
  addRule_(slide, 40, 96, 640, COLORS.gold);
}

function addCoverSlide_(slide, data) {
  setSlideBackground_(slide, COLORS.navy);
  addText_(slide, data.deck_title, 54, 125, 610, 58, {
    fontFamily: 'Aptos Display',
    fontSize: 46,
    color: COLORS.white,
    bold: true,
  });
  addRule_(slide, 56, 198, 120, COLORS.gold);
  addText_(slide, 'Compression results', 56, 217, 580, 30, {
    fontSize: 22,
    color: '#D7E2EC',
  });
  addText_(slide, data.dataset, 56, 336, 500, 18, {
    fontSize: 12,
    color: '#B2C5D5',
  });
}

function addBaselineSlide_(slide, data) {
  setSlideBackground_(slide, COLORS.paper);
  addSlideHeading_(slide, 'Baseline and quantization check',
    'All comparison slides use the original FP32 Baseline as their reference');
  addTable_(slide, [
    ['Metric', 'Baseline (FP32)', 'Baseline (quantized)'],
    ...METRICS.map(function(metric) {
      return [metric[0], formatMetric_(metric[1], data.baseline.fp32), formatMetric_(metric[1], data.baseline.quantized)];
    }),
  ], 40, 112, 640, 175, 11.5);
  const range = quantizationAccuracyRange_(data);
  addText_(slide,
    'Across all 15 models, quantization changed test accuracy from ' +
    signed_(range.min) + ' pp to ' + signed_(range.max) + ' pp.',
    56, 310, 608, 28, {
      fontSize: 15,
      color: COLORS.ink,
      align: SlidesApp.ParagraphAlignment.CENTER,
    }
  );
  addFooter_(slide, 'Quantized results use INT8 supported layers with FP16 fallback.');
}

function addGhostFp32Slide_(slide, data) {
  setSlideBackground_(slide, COLORS.paper);
  addSlideHeading_(slide, 'Ghost Convolution results', 'FP32 absolute metrics and comparison with the FP32 Baseline');
  const baseline = data.baseline.fp32;
  const models = [
    ['Baseline', baseline],
    ['All kernel sizes', experimentMetrics_(data, 'ghost_allk', 'fp32')],
    ['Kernel size = 1', experimentMetrics_(data, 'ghost_k1', 'fp32')],
    ['Kernel size > 1', experimentMetrics_(data, 'ghost_gt1', 'fp32')],
  ];
  addText_(slide, 'Absolute metrics', 40, 108, 280, 17, { fontSize: 12, color: COLORS.muted, bold: true });
  addTable_(slide, absoluteRows_(models), 40, 128, 640, 112, 10.5);
  addText_(slide, 'Change from Baseline', 40, 252, 370, 17, { fontSize: 12, color: COLORS.muted, bold: true });
  addTable_(slide, comparisonRows_(models.slice(1), baseline), 40, 272, 640, 85, 9.5);
  addFooter_(slide, 'Accuracy uses percentage points. Resource metrics use relative percentages.');
}

function addGhostQuantizedComparisonSlide_(slide, data) {
  setSlideBackground_(slide, COLORS.paper);
  addSlideHeading_(slide, 'Ghost Convolution after quantization',
    'Quantized Ghost models: comparison with the FP32 Baseline');
  const baseline = data.baseline.fp32;
  const models = [
    ['All kernel sizes', experimentMetrics_(data, 'ghost_allk', 'quantized')],
    ['Kernel size = 1', experimentMetrics_(data, 'ghost_k1', 'quantized')],
    ['Kernel size > 1', experimentMetrics_(data, 'ghost_gt1', 'quantized')],
  ];
  addTable_(slide, comparisonRows_(models, baseline), 40, 122, 640, 165, 11);
  addFooter_(slide, 'Accuracy uses percentage points. Resource metrics use relative percentages. Reference: FP32 Baseline.');
}

function addKDAbsoluteSlide_(slide, data) {
  setSlideBackground_(slide, COLORS.paper);
  addSlideHeading_(slide, 'Knowledge distillation results', 'FP32 absolute metrics for Ghost models and their KD-recovered versions');
  const baseline = data.baseline.fp32;
  const models = [
    ['Baseline', baseline],
    ['All kernel sizes, Ghost', experimentMetrics_(data, 'ghost_allk', 'fp32')],
    ['All kernel sizes, KD', experimentMetrics_(data, 'kd_ghost_allk', 'fp32')],
    ['Kernel size = 1, Ghost', experimentMetrics_(data, 'ghost_k1', 'fp32')],
    ['Kernel size = 1, KD', experimentMetrics_(data, 'kd_ghost_k1', 'fp32')],
    ['Kernel size > 1, Ghost', experimentMetrics_(data, 'ghost_gt1', 'fp32')],
    ['Kernel size > 1, KD', experimentMetrics_(data, 'kd_ghost_gt1', 'fp32')],
  ];
  addTable_(slide, absoluteRows_(models), 40, 112, 640, 235, 10.5);
  addFooter_(slide, 'KD begins from the matching Ghost model and uses the FP32 Baseline as teacher.');
}

function addKDFp32ComparisonSlide_(slide, data) {
  setSlideBackground_(slide, COLORS.paper);
  addSlideHeading_(slide, 'Knowledge distillation recovery', 'FP32 comparison with the Baseline and the matching Ghost model');
  const baseline = data.baseline.fp32;
  const pairs = [
    ['All', experimentMetrics_(data, 'kd_ghost_allk', 'fp32'), experimentMetrics_(data, 'ghost_allk', 'fp32')],
    ['K = 1', experimentMetrics_(data, 'kd_ghost_k1', 'fp32'), experimentMetrics_(data, 'ghost_k1', 'fp32')],
    ['K > 1', experimentMetrics_(data, 'kd_ghost_gt1', 'fp32'), experimentMetrics_(data, 'ghost_gt1', 'fp32')],
  ];
  addKDComparisonTable_(slide, pairs, baseline);
  addFooter_(slide, 'Positive accuracy change versus Ghost indicates KD recovery. Resource values remain effectively unchanged.');
}

function addKDQuantizedComparisonSlide_(slide, data) {
  setSlideBackground_(slide, COLORS.paper);
  addSlideHeading_(slide, 'Knowledge distillation after quantization',
    'Quantized KD models: comparison with the FP32 Baseline');
  const baseline = data.baseline.fp32;
  const models = [
    ['All kernel sizes', experimentMetrics_(data, 'kd_ghost_allk', 'quantized')],
    ['Kernel size = 1', experimentMetrics_(data, 'kd_ghost_k1', 'quantized')],
    ['Kernel size > 1', experimentMetrics_(data, 'kd_ghost_gt1', 'quantized')],
  ];
  addTable_(slide, comparisonRows_(models, baseline), 40, 122, 640, 165, 11);
  addFooter_(slide, 'Accuracy uses percentage points. Resource metrics use relative percentages. Reference: FP32 Baseline.');
}

function addLowRankAbsoluteSlide_(slide, data, target) {
  setSlideBackground_(slide, COLORS.paper);
  const title = target === 'transformer'
    ? 'Low-rank factorization: Transformer layers'
    : 'Low-rank factorization: all target layers';
  addSlideHeading_(slide, title, 'FP32 absolute metrics across four rank ratios');
  const baseline = data.baseline.fp32;
  const models = [['Baseline', baseline]];
  rankRows_(data, target, 'fp32').forEach(function(item) { models.push(item); });
  addTable_(slide, absoluteRows_(models), 40, 115, 640, 195, 11);
  addFooter_(slide, 'All target layers: Transformer, convolution, and projection layers.');
}

function addLowRankComparisonSlide_(slide, data, target, precision) {
  setSlideBackground_(slide, COLORS.paper);
  const targetLabel = target === 'transformer' ? 'Transformer layers' : 'all target layers';
  const precisionLabel = precision === 'fp32' ? 'FP32' : 'quantized';
  addSlideHeading_(slide, 'Low-rank ' + targetLabel + ' changes',
    precisionLabel + ' models: comparison with the FP32 Baseline');
  addTable_(slide, comparisonRows_(rankRows_(data, target, precision), data.baseline.fp32),
    40, 120, 640, 175, 11);
  addFooter_(slide, 'Accuracy uses percentage points. Resource metrics use relative percentages.');
}

function addGhostSummarySlide_(slide, data) {
  setSlideBackground_(slide, COLORS.paper);
  addSlideHeading_(slide, 'Ghost Convolution summary', 'All FP32 Ghost variants relative to the FP32 Baseline');
  const baseline = data.baseline.fp32;
  const models = [
    ['All kernel sizes', experimentMetrics_(data, 'ghost_allk', 'fp32')],
    ['Kernel size = 1', experimentMetrics_(data, 'ghost_k1', 'fp32')],
    ['Kernel size > 1', experimentMetrics_(data, 'ghost_gt1', 'fp32')],
  ];
  addTable_(slide, summaryComparisonRows_(models, baseline), 40, 122, 640, 165, 11);
  addFooter_(slide, 'FP32 results only. Accuracy uses percentage points. Resource metrics use relative percentages.');
}

function addKDSummarySlide_(slide, data) {
  setSlideBackground_(slide, COLORS.paper);
  addSlideHeading_(slide, 'Knowledge Distillation summary', 'All FP32 KD variants relative to the FP32 Baseline');
  const baseline = data.baseline.fp32;
  const models = [
    ['All kernel sizes', experimentMetrics_(data, 'kd_ghost_allk', 'fp32')],
    ['Kernel size = 1', experimentMetrics_(data, 'kd_ghost_k1', 'fp32')],
    ['Kernel size > 1', experimentMetrics_(data, 'kd_ghost_gt1', 'fp32')],
  ];
  addTable_(slide, summaryComparisonRows_(models, baseline), 40, 122, 640, 165, 11);
  addFooter_(slide, 'FP32 results only. Accuracy uses percentage points. Resource metrics use relative percentages.');
}

function addLowRankSummarySlide_(slide, data) {
  setSlideBackground_(slide, COLORS.paper);
  addSlideHeading_(slide, 'Low-Rank Factorization summary', 'All FP32 low-rank variants relative to the FP32 Baseline');
  const models = [
    ['Transformer, r = 0.125', experimentMetrics_(data, 'lowrank_transformer_r0125', 'fp32')],
    ['Transformer, r = 0.25', experimentMetrics_(data, 'lowrank_transformer_r25', 'fp32')],
    ['Transformer, r = 0.50', experimentMetrics_(data, 'lowrank_transformer_r05', 'fp32')],
    ['Transformer, r = 0.75', experimentMetrics_(data, 'lowrank_transformer_r075', 'fp32')],
    ['All target layers, r = 0.125', experimentMetrics_(data, 'lowrank_all_r0125', 'fp32')],
    ['All target layers, r = 0.25', experimentMetrics_(data, 'lowrank_all_r25', 'fp32')],
    ['All target layers, r = 0.50', experimentMetrics_(data, 'lowrank_all_r05', 'fp32')],
    ['All target layers, r = 0.75', experimentMetrics_(data, 'lowrank_all_r075', 'fp32')],
  ];
  addTable_(slide, summaryComparisonRows_(models, data.baseline.fp32), 40, 112, 640, 245, 10);
  addFooter_(slide, 'FP32 results only. Accuracy uses percentage points. Resource metrics use relative percentages.');
}

function rankRows_(data, target, precision) {
  return [
    ['r = 0.125', experimentMetrics_(data, 'lowrank_' + target + '_r0125', precision)],
    ['r = 0.25', experimentMetrics_(data, 'lowrank_' + target + '_r25', precision)],
    ['r = 0.50', experimentMetrics_(data, 'lowrank_' + target + '_r05', precision)],
    ['r = 0.75', experimentMetrics_(data, 'lowrank_' + target + '_r075', precision)],
  ];
}

function experimentMetrics_(data, idPrefix, precision) {
  const id = idPrefix + '_' + precision;
  const item = data.experiments.find(function(experiment) { return experiment.id === id; });
  if (!item) throw new Error('Missing experiment: ' + id);
  return item.metrics;
}

function absoluteRows_(models) {
  const rows = [['Model', 'Accuracy', 'Parameters', 'Latency', 'Model size', 'FLOPs']];
  models.forEach(function(model) {
    rows.push([model[0]].concat(METRICS.map(function(metric) {
      return formatMetric_(metric[1], model[1]);
    })));
  });
  return rows;
}

function comparisonRows_(models, baseline) {
  const rows = [['Model', 'Accuracy', 'Parameters', 'Latency', 'Model size', 'FLOPs']];
  models.forEach(function(model) {
    rows.push([model[0]].concat(METRICS.map(function(metric) {
      return formatComparison_(metric[1], model[1], baseline);
    })));
  });
  return rows;
}

function summaryComparisonRows_(models, baseline) {
  const rows = [['Model', 'Accuracy', 'Parameters', 'Latency', 'Model size', 'FLOPs']];
  models.forEach(function(model) {
    rows.push([
      model[0],
      formatComparison_('accuracy', model[1], baseline) + ' (' + formatMetric_('accuracy', model[1]) + ')',
      formatComparison_('param_count_tensors', model[1], baseline),
      formatComparison_('latency_ms_per_batch', model[1], baseline),
      formatComparison_('model_size_mb', model[1], baseline),
      formatComparison_('flops_per_batch', model[1], baseline),
    ]);
  });
  return rows;
}

function addKDComparisonTable_(slide, pairs, baseline) {
  const rows = [['Metric', 'All vs\nBaseline', 'All vs\nGhost', 'K = 1 vs\nBaseline', 'K = 1 vs\nGhost', 'K > 1 vs\nBaseline', 'K > 1 vs\nGhost']];
  METRICS.forEach(function(metric) {
    const row = [metric[0]];
    pairs.forEach(function(pair) {
      row.push(formatComparison_(metric[1], pair[1], baseline));
      row.push(formatComparison_(metric[1], pair[1], pair[2]));
    });
    rows.push(row);
  });
  addTable_(slide, rows, 30, 118, 660, 220, 9);
}

function formatMetric_(metric, values) {
  const value = values[metric];
  switch (metric) {
    case 'accuracy': return (value * 100).toFixed(2) + '%';
    case 'param_count_tensors': return (value / 1e6).toFixed(2) + ' M';
    case 'latency_ms_per_batch': return value.toFixed(2) + ' ms';
    case 'model_size_mb': return value.toFixed(2) + ' MB';
    case 'flops_per_batch': return (value / 1e9).toFixed(2) + ' G';
    default: throw new Error('Unknown metric: ' + metric);
  }
}

function formatComparison_(metric, model, baseline) {
  if (metric === 'accuracy') {
    return signed_((model.accuracy - baseline.accuracy) * 100) + ' pp';
  }
  return signed_(((model[metric] - baseline[metric]) / baseline[metric]) * 100) + '%';
}

function quantizationAccuracyRange_(data) {
  const changes = [(data.baseline.quantized.accuracy - data.baseline.fp32.accuracy) * 100];
  data.experiments.forEach(function(experiment) {
    if (!experiment.id.endsWith('_quantized')) return;
    const fp32Id = experiment.id.replace('_quantized', '_fp32');
    const fp32 = data.experiments.find(function(item) { return item.id === fp32Id; });
    if (fp32) changes.push((experiment.metrics.accuracy - fp32.metrics.accuracy) * 100);
  });
  return { min: Math.min.apply(null, changes), max: Math.max.apply(null, changes) };
}

function addTable_(slide, rows, x, y, width, height, fontSize) {
  const table = slide.insertTable(rows.length, rows[0].length, x, y, width, height);
  for (let row = 0; row < rows.length; row += 1) {
    for (let column = 0; column < rows[row].length; column += 1) {
      const cell = table.getCell(row, column);
      cell.getText().setText(rows[row][column]);
      const style = cell.getText().getTextStyle();
      style.setFontFamily('Aptos');
      style.setFontSize(row === 0 ? Math.max(fontSize - 0.5, 8) : fontSize);
      style.setBold(row === 0 || column === 0);
      style.setForegroundColor(row === 0 ? COLORS.white : COLORS.ink);
      cell.getText().getParagraphStyle().setParagraphAlignment(
        column === 0 ? SlidesApp.ParagraphAlignment.START : SlidesApp.ParagraphAlignment.CENTER
      );
      cell.getFill().setSolidFill(row === 0 ? COLORS.ink : (row % 2 === 0 ? COLORS.white : '#EDF2F5'));
    }
  }
  return table;
}

function signed_(value) {
  return (value >= 0 ? '+' : '') + value.toFixed(2);
}
