import 'package:trajectory_app/models/record_model.dart';

class ChartPointModel {
  final DateTime date; // 用來做 X 軸（時間軸）
  final double value; // Y 軸的數值，例如腦齡或實際年齡

  ChartPointModel({required this.date, required this.value});
}

Map<String, List<ChartPointModel>> extractChartPoints(
  List<RecordModel> records,
) {
  final actualAgePoints = <ChartPointModel>[];
  final brainAgePoints = <ChartPointModel>[];

  for (var r in records) {
    try {
      final date = DateTime.parse('${r.yyyy}${r.mm}${r.dd}');

      final actualAge = double.tryParse(r.actualAge);
      if (actualAge != null) {
        actualAgePoints.add(ChartPointModel(date: date, value: actualAge));
      }

      final brainAge = double.tryParse(r.brainAge);
      if (brainAge != null) {
        brainAgePoints.add(ChartPointModel(date: date, value: brainAge));
      }
    } catch (_) {
      continue;
    }
  }

  // ⭐️ 可選擇性排序（讓折線圖更穩定）
  actualAgePoints.sort((a, b) => a.date.compareTo(b.date));
  brainAgePoints.sort((a, b) => a.date.compareTo(b.date));

  return {'actual': actualAgePoints, 'brain': brainAgePoints};
}
