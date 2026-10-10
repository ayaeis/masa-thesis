/*
 * Append a two-slide MSASL100 low-rank KD section to an existing Google Slides deck.
 *
 * 1. Upload msasl100_lowrank_kd_slides_data.json to Google Drive.
 * 2. Set LRKD_DATA_FILE_ID to its Drive file ID.
 * 3. Set LRKD_PRESENTATION_ID to the ID in your existing MSASL100 Slides URL.
 * 4. Run appendMSASL100LowRankKDSlides().
 */

const LRKD_DATA_FILE_ID = 'PASTE_GOOGLE_DRIVE_JSON_FILE_ID_HERE';
const LRKD_PRESENTATION_ID = 'PASTE_EXISTING_MSASL100_PRESENTATION_ID_HERE';

const LRKD_COLORS = {
  ink: '#102A43',
  paper: '#F7F4EE',
  muted: '#5D6D7E',
  gold: '#D09A36',
};

function appendMSASL100LowRankKDSlides() {
  if (LRKD_DATA_FILE_ID === 'PASTE_GOOGLE_DRIVE_JSON_FILE_ID_HERE') {
    throw new Error('Set LRKD_DATA_FILE_ID to the uploaded JSON file ID.');
  }
  if (LRKD_PRESENTATION_ID === 'PASTE_EXISTING_MSASL100_PRESENTATION_ID_HERE') {
    throw new Error('Set LRKD_PRESENTATION_ID to your existing MSASL100 deck ID.');
  }

  const data = JSON.parse(
    DriveApp.getFileById(LRKD_DATA_FILE_ID).getBlob().getDataAsString('UTF-8')
  );
  lrkdValidateData_(data);

  const deck = SlidesApp.openById(LRKD_PRESENTATION_ID);
  lrkdAddSetupSlide_(deck.appendSlide(SlidesApp.PredefinedLayout.BLANK), data);
  lrkdAddResultsSlide_(deck.appendSlide(SlidesApp.PredefinedLayout.BLANK), data);
  Logger.log('Added 2 slides: ' + deck.getUrl());
  return deck.getUrl();
}

function lrkdValidateData_(data) {
  if (!data || !data.setup || !Array.isArray(data.experiments) || data.experiments.length !== 3) {
    throw new Error('Unexpected MSASL100 low-rank KD JSON schema.');
  }
}

function lrkdAddSetupSlide_(slide, data) {
  lrkdSetBackground_(slide, LRKD_COLORS.paper);
  lrkdAddHeading_(slide, 'Low-rank KD recovery', 'Targeted MSASL100 ablation');
  const setup = data.setup;
  lrkdAddText_(slide, 'Question', 52, 122, 120, 22, {fontSize: 17, bold: true});
  lrkdAddText_(slide,
    'Does knowledge distillation improve post-training low-rank models that already retain strong accuracy?',
    185, 119, 455, 42, {fontSize: 18});
  lrkdAddText_(slide, 'Teacher', 52, 180, 120, 22, {fontSize: 17, bold: true});
  lrkdAddText_(slide, setup.teacher, 185, 180, 455, 22, {fontSize: 18});
  lrkdAddText_(slide, 'Student', 52, 222, 120, 22, {fontSize: 17, bold: true});
  lrkdAddText_(slide,
    setup.student + ', all target layers, r = 0.125 / 0.25 / 0.50',
    185, 222, 455, 42, {fontSize: 18});
  lrkdAddText_(slide, 'KD recovery', 52, 282, 120, 22, {fontSize: 17, bold: true});
  lrkdAddText_(slide,
    setup.epochs + ' epochs, alpha = ' + setup.kd_alpha + ', temperature = ' + setup.kd_temperature,
    185, 282, 455, 22, {fontSize: 18});
  lrkdAddText_(slide,
    'KD changes the model weights only. The low-rank architecture and its parameter and FLOP counts remain unchanged.',
    52, 342, 590, 28, {fontSize: 13, color: LRKD_COLORS.muted});
}

function lrkdAddResultsSlide_(slide, data) {
  lrkdSetBackground_(slide, LRKD_COLORS.paper);
  lrkdAddHeading_(slide, 'Low-rank KD recovery results', 'Pre-KD post-training student compared with its KD-recovered version');
  const rows = [['Rank', 'Stage', 'Accuracy', 'Parameters', 'Latency', 'Model size', 'FLOPs']];
  data.experiments.forEach(function(item) {
    rows.push([lrkdRankLabel_(item.rank), 'Pre-KD', ...lrkdMetricCells_(item.pre_kd_metrics)]);
    rows.push([lrkdRankLabel_(item.rank), 'After KD', ...lrkdMetricCells_(item.kd_metrics)]);
  });
  lrkdAddTable_(slide, rows, 28, 112, 664, 180, 10);

  const recoveryText = data.experiments.map(function(item) {
    return 'r = ' + item.rank + ': ' + lrkdSignedPp_(item.delta_vs_pre_kd.accuracy_pp);
  }).join('   ');
  lrkdAddText_(slide, 'Accuracy change after KD: ' + recoveryText,
    45, 314, 625, 24, {fontSize: 16, bold: true});
  lrkdAddText_(slide,
    'Accuracy change uses percentage points. Latency is a separate timing measurement and may vary slightly despite identical architecture.',
    45, 350, 625, 22, {fontSize: 11, color: LRKD_COLORS.muted});
}

function lrkdRankLabel_(rank) {
  return 'r = ' + rank;
}

function lrkdMetricCells_(metrics) {
  return [
    (metrics.accuracy * 100).toFixed(2) + '%',
    (metrics.param_count_tensors / 1e6).toFixed(2) + ' M',
    metrics.latency_ms_per_batch.toFixed(2) + ' ms',
    metrics.model_size_mb.toFixed(2) + ' MB',
    (metrics.flops_per_batch / 1e9).toFixed(2) + ' G',
  ];
}

function lrkdSignedPp_(value) {
  return (value >= 0 ? '+' : '') + value.toFixed(2) + ' pp';
}

function lrkdSetBackground_(slide, color) {
  slide.getBackground().setSolidFill(color);
}

function lrkdAddHeading_(slide, title, subtitle) {
  lrkdAddText_(slide, title, 40, 26, 640, 33, {
    fontFamily: 'Aptos Display', fontSize: 27, color: LRKD_COLORS.ink, bold: true,
  });
  lrkdAddText_(slide, subtitle, 40, 66, 640, 20, {
    fontSize: 12.5, color: LRKD_COLORS.muted,
  });
  const line = slide.insertLine(SlidesApp.LineCategory.STRAIGHT, 40, 96, 680, 96);
  line.getLineFill().setSolidFill(LRKD_COLORS.gold);
  line.setWeight(1.4);
}

function lrkdAddText_(slide, text, x, y, width, height, options) {
  const settings = options || {};
  const shape = slide.insertTextBox(text, x, y, width, height);
  const range = shape.getText();
  range.getTextStyle()
    .setFontFamily(settings.fontFamily || 'Aptos')
    .setFontSize(settings.fontSize || 18)
    .setForegroundColor(settings.color || LRKD_COLORS.ink)
    .setBold(Boolean(settings.bold));
  shape.getFill().setTransparent();
  return shape;
}

function lrkdAddTable_(slide, rows, x, y, width, height, fontSize) {
  const table = slide.insertTable(rows.length, rows[0].length, x, y, width, height);
  rows.forEach(function(row, rowIndex) {
    row.forEach(function(value, colIndex) {
      const cell = table.getCell(rowIndex, colIndex);
      cell.getText().setText(value);
      cell.getText().getTextStyle()
        .setFontFamily('Aptos')
        .setFontSize(fontSize)
        .setForegroundColor(rowIndex === 0 ? '#FFFFFF' : LRKD_COLORS.ink)
        .setBold(rowIndex === 0);
      cell.getFill().setSolidFill(rowIndex === 0 ? LRKD_COLORS.ink : '#FFFFFF');
    });
  });
}
