/*
 * MSASL100 post-training versus architecture-first comparison deck.
 *
 * Setup:
 * 1. Upload msasl100_architecture_first_slides_data.json to Google Drive.
 * 2. Put its Drive file ID in DATA_FILE_ID.
 * 3. Open script.new, paste this file, and run createMSASL100ArchitectureFirstDeck().
 */

const DATA_FILE_ID = 'PASTE_GOOGLE_DRIVE_JSON_FILE_ID_HERE';

const COLORS = {
  ink: '#102A43',
  navy: '#0B1F33',
  paper: '#F7F4EE',
  white: '#FFFFFF',
  muted: '#5D6D7E',
  gold: '#D09A36',
  green: '#217A55',
  red: '#B13A38',
};

const METRICS = [
  ['Accuracy', 'accuracy'],
  ['Parameters', 'param_count_tensors'],
  ['Latency', 'latency_ms_per_batch'],
  ['Model size', 'model_size_mb'],
  ['FLOPs', 'flops_per_batch'],
];

function createMSASL100ArchitectureFirstDeck() {
  if (DATA_FILE_ID === 'PASTE_GOOGLE_DRIVE_JSON_FILE_ID_HERE') {
    throw new Error('Set DATA_FILE_ID to the uploaded JSON file ID first.');
  }

  const data = JSON.parse(
    DriveApp.getFileById(DATA_FILE_ID).getBlob().getDataAsString('UTF-8')
  );
  validateData_(data);

  const deck = SlidesApp.create(data.deck_title);
  const first = deck.getSlides()[0];
  clearSlide_(first);
  addCover_(first, data);
  addProtocol_(blank_(deck), data);
  addAbsolute_(blank_(deck), data, 'ghost', 'Ghost Convolution: all kernel sizes');
  addComparison_(blank_(deck), data, 'ghost', 'Ghost Convolution: change from baseline');
  addAbsolute_(blank_(deck), data, 'low_rank', 'Low-rank factorization: all targets, r = 0.125');
  addComparison_(blank_(deck), data, 'low_rank', 'Low-rank factorization: change from baseline');
  addConclusion_(blank_(deck), data);

  Logger.log('Created 7 slides: ' + deck.getUrl());
  return deck.getUrl();
}

function validateData_(data) {
  if (!data || !data.baseline || !data.methods || !data.methods.ghost || !data.methods.low_rank) {
    throw new Error('Unexpected architecture-first comparison JSON schema.');
  }
  if (data.methods.ghost.length !== 3 || data.methods.low_rank.length !== 3) {
    throw new Error('Expected post-training, architecture-first 60, and architecture-first 120 entries.');
  }
}

function blank_(deck) {
  return deck.appendSlide(SlidesApp.PredefinedLayout.BLANK);
}

function clearSlide_(slide) {
  slide.getPageElements().forEach(function(element) { element.remove(); });
}

function background_(slide, color) {
  slide.getBackground().setSolidFill(color);
}

function text_(slide, value, x, y, width, height, options) {
  const settings = options || {};
  const shape = slide.insertTextBox(value, x, y, width, height);
  const range = shape.getText();
  range.getTextStyle()
    .setFontFamily(settings.fontFamily || 'Aptos')
    .setFontSize(settings.fontSize || 18)
    .setForegroundColor(settings.color || COLORS.ink)
    .setBold(Boolean(settings.bold));
  range.getParagraphStyle().setParagraphAlignment(
    settings.align || SlidesApp.ParagraphAlignment.START
  );
  shape.getFill().setTransparent();
  return shape;
}

function rule_(slide, x, y, width) {
  const line = slide.insertLine(SlidesApp.LineCategory.STRAIGHT, x, y, x + width, y);
  line.getLineFill().setSolidFill(COLORS.gold);
  line.setWeight(1.5);
}

function footer_(slide, value) {
  text_(slide, value, 40, 377, 640, 12, {
    fontSize: 8.5,
    color: COLORS.muted,
    align: SlidesApp.ParagraphAlignment.CENTER,
  });
}

function heading_(slide, title, subtitle) {
  text_(slide, title, 40, 26, 640, 34, {
    fontFamily: 'Aptos Display', fontSize: 27, color: COLORS.ink, bold: true,
  });
  text_(slide, subtitle, 40, 66, 640, 20, { fontSize: 12.5, color: COLORS.muted });
  rule_(slide, 40, 96, 640);
}

function addCover_(slide, data) {
  background_(slide, COLORS.navy);
  text_(slide, 'MSASL100', 56, 113, 580, 48, {
    fontFamily: 'Aptos Display', fontSize: 44, color: COLORS.white, bold: true,
  });
  text_(slide, 'Post-training vs architecture-first compression', 56, 171, 590, 35, {
    fontSize: 23, color: '#D7E2EC',
  });
  rule_(slide, 56, 220, 140);
  text_(slide, data.dataset, 56, 338, 540, 18, { fontSize: 12, color: '#B2C5D5' });
}

function addProtocol_(slide, data) {
  background_(slide, COLORS.paper);
  heading_(slide, 'Comparison protocol', 'The architecture sequence is the experimental variable');
  const rows = [
    ['Condition', 'Training sequence'],
    ['Post-training', data.protocol.post_training],
    ['Architecture-first, 60', data.protocol.architecture_first_60],
    ['Architecture-first, two stages', data.protocol.architecture_first_120],
  ];
  table_(slide, rows, 40, 116, 640, 170, 12);
  text_(slide, 'Controlled settings: ' + data.protocol.controlled, 54, 315, 612, 34, {
    fontSize: 14, color: COLORS.ink, align: SlidesApp.ParagraphAlignment.CENTER,
  });
  footer_(slide, 'All values are FP32 and measured against the same MSASL100 baseline.');
}

function addAbsolute_(slide, data, method, title) {
  background_(slide, COLORS.paper);
  heading_(slide, title, 'Absolute FP32 metrics');
  const models = [['Baseline', data.baseline]];
  data.methods[method].forEach(function(item) { models.push([item.label, item.metrics]); });
  table_(slide, absoluteRows_(models), 25, 122, 670, 170, 11);
  footer_(slide, 'Parameters and FLOPs describe the model architecture. Accuracy and latency reflect training and measurement.');
}

function addComparison_(slide, data, method, title) {
  background_(slide, COLORS.paper);
  heading_(slide, title, 'Positive resource values mean a reduction relative to the FP32 baseline');
  table_(slide, changeRows_(data.methods[method], data.baseline), 25, 122, 670, 155, 10.5);
  const post = data.methods[method][0].metrics;
  const af120 = data.methods[method][2].metrics;
  const accuracyDelta = (af120.accuracy - post.accuracy) * 100;
  const color = accuracyDelta >= 0 ? COLORS.green : COLORS.red;
  text_(slide,
    'Architecture-first, two stages vs post-training: ' + signed_(accuracyDelta) + ' pp accuracy.',
    45, 315, 630, 24, {
      fontSize: 17, color: color, bold: true, align: SlidesApp.ParagraphAlignment.CENTER,
    }
  );
  footer_(slide, 'Accuracy uses percentage points. Parameter, FLOP, size, and latency values are savings versus baseline.');
}

function addConclusion_(slide, data) {
  background_(slide, COLORS.paper);
  heading_(slide, 'Results summary', 'Architecture-first training affects accuracy, not the compression structure');
  const ghost = data.methods.ghost;
  const lowRank = data.methods.low_rank;
  const rows = [
    ['Method', 'Post-training', 'Architecture-first, 60', 'Architecture-first, two stages'],
    ['Ghost Convolution', accuracy_(ghost[0].metrics), accuracy_(ghost[1].metrics), accuracy_(ghost[2].metrics)],
    ['Low-rank factorization', accuracy_(lowRank[0].metrics), accuracy_(lowRank[1].metrics), accuracy_(lowRank[2].metrics)],
  ];
  table_(slide, rows, 40, 120, 640, 105, 13);
  text_(slide,
    'Ghost Conv performed best when the compressed architecture learned MSASL100 from the start. ' +
    'Low-rank factorization performed best after dense task-specific fine-tuning and recovery.',
    58, 260, 585, 48, { fontSize: 17, color: COLORS.ink, align: SlidesApp.ParagraphAlignment.CENTER }
  );
  text_(slide,
    'The low-rank configuration retains its 40.27% parameter saving and 79.99% FLOP saving under either training path.',
    58, 329, 585, 24, { fontSize: 13, color: COLORS.muted, align: SlidesApp.ParagraphAlignment.CENTER }
  );
  footer_(slide, 'Single-seed controlled comparison. Repeat across additional seeds for statistical confidence.');
}

function absoluteRows_(models) {
  const rows = [['Model', 'Accuracy', 'Parameters', 'Latency', 'Model size', 'FLOPs']];
  models.forEach(function(model) {
    rows.push([model[0]].concat(METRICS.map(function(metric) {
      return metricValue_(metric[1], model[1]);
    })));
  });
  return rows;
}

function changeRows_(models, baseline) {
  const rows = [['Model', 'Accuracy', 'Parameter saving', 'FLOP saving', 'Size saving', 'Latency reduction']];
  models.forEach(function(item) {
    const m = item.metrics;
    rows.push([
      item.label,
      signed_((m.accuracy - baseline.accuracy) * 100) + ' pp',
      saving_('param_count_tensors', m, baseline),
      saving_('flops_per_batch', m, baseline),
      saving_('model_size_mb', m, baseline),
      saving_('latency_ms_per_batch', m, baseline),
    ]);
  });
  return rows;
}

function metricValue_(metric, values) {
  const value = values[metric];
  if (metric === 'accuracy') return (value * 100).toFixed(2) + '%';
  if (metric === 'param_count_tensors') return (value / 1e6).toFixed(2) + ' M';
  if (metric === 'latency_ms_per_batch') return value.toFixed(2) + ' ms';
  if (metric === 'model_size_mb') return value.toFixed(2) + ' MB';
  if (metric === 'flops_per_batch') return (value / 1e9).toFixed(2) + ' G';
  throw new Error('Unknown metric: ' + metric);
}

function accuracy_(metrics) {
  return (metrics.accuracy * 100).toFixed(2) + '%';
}

function saving_(metric, model, baseline) {
  return (100 * (1 - model[metric] / baseline[metric])).toFixed(2) + '%';
}

function signed_(value) {
  return (value >= 0 ? '+' : '') + value.toFixed(2);
}

function table_(slide, rows, x, y, width, height, fontSize) {
  const table = slide.insertTable(rows.length, rows[0].length, x, y, width, height);
  for (let row = 0; row < rows.length; row += 1) {
    for (let column = 0; column < rows[row].length; column += 1) {
      const cell = table.getCell(row, column);
      cell.getText().setText(rows[row][column]);
      const style = cell.getText().getTextStyle();
      style.setFontFamily('Aptos');
      style.setFontSize(row === 0 ? Math.max(8, fontSize - 0.5) : fontSize);
      style.setBold(row === 0 || column === 0);
      style.setForegroundColor(row === 0 ? COLORS.white : COLORS.ink);
      cell.getFill().setSolidFill(row === 0 ? COLORS.navy : (row % 2 ? COLORS.white : '#EDF1F4'));
    }
  }
  return table;
}
