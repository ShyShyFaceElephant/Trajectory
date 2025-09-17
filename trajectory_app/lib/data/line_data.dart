import 'package:fl_chart/fl_chart.dart';

class LineData {
  final List<FlSpot> line1 = const [
    FlSpot(2010, 50),
    FlSpot(2015, 55),
    FlSpot(2018, 58),
    FlSpot(2022, 62),
  ];
  final List<FlSpot> line2 = const [
    FlSpot(2010, 50),
    FlSpot(2015, 56),
    FlSpot(2018, 59),
    FlSpot(2022, 71),
  ];

  final leftTitle = {
    45: '45',
    50: '50',
    55: '55',
    60: '60',
    65: '65',
    70: '70',
    75: '75',
  };

  final bottomTitle = {2010: '2010', 2015: '2015', 2018: '2018', 2022: '2022'};
}
