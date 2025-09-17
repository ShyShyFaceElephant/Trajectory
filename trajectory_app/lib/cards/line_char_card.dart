import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:trajectory_app/cards/custom_card.dart';
import 'package:trajectory_app/models/chart_point_model.dart';

class LineChartCard extends StatelessWidget {
  final List<ChartPointModel> brainAgePoints;
  final List<ChartPointModel> actualAgePoints;

  const LineChartCard({
    super.key,
    required this.brainAgePoints,
    required this.actualAgePoints,
  });

  @override
  Widget build(BuildContext context) {
    if (brainAgePoints.isEmpty) {
      return const CustomCard(
        child: Padding(
          padding: EdgeInsets.symmetric(horizontal: 15, vertical: 10),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Text(
                    '腦齡變化趨勢',
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.w500),
                  ),
                  SizedBox(width: 30),
                  Text(
                    '● 實際年齡',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                      color: Colors.greenAccent,
                    ),
                  ),
                  SizedBox(width: 30),
                  Text(
                    '● 腦部年齡',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                      color: Colors.orange,
                    ),
                  ),
                ],
              ),
              SizedBox(height: 20),
              Center(
                child: Padding(
                  padding: EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                  child: Text(
                    "影像紀錄小於 2 次，無法繪製趨勢圖。",
                    style: TextStyle(fontSize: 14, color: Colors.white70),
                  ),
                ),
              ),
            ],
          ),
        ),
      );
    }
    final allPoints = [...brainAgePoints, ...actualAgePoints]
      ..sort((a, b) => a.date.compareTo(b.date));
    // 動態設定 Y 軸最大值（下一個 5 的倍數）
    final allValues = allPoints.map((e) => e.value);
    final minY = (allValues.reduce((a, b) => a < b ? a : b) / 5).floor() * 5.0;
    final maxY = (allValues.reduce((a, b) => a > b ? a : b) / 5).ceil() * 5.0;
    if (allPoints.isEmpty) {
      return const Center(child: Text("沒有可顯示的資料"));
    }

    final baseDate = DateTime(allPoints.first.date.year, 1, 1);
    final endDate = DateTime(allPoints.last.date.year + 1, 1, 1);
    final maxOffset = endDate.difference(baseDate).inDays.toDouble();
    int intervalWeight;
    if ((endDate.year - baseDate.year) > 40) {
      intervalWeight = 3;
    } else if ((endDate.year - baseDate.year) > 20) {
      intervalWeight = 2;
    } else {
      intervalWeight = 1;
    }
    int dateToOffset(DateTime d) => d.difference(baseDate).inDays;

    List<FlSpot> toSpots(List<ChartPointModel> list) {
      return list
          .map((p) => FlSpot(dateToOffset(p.date).toDouble(), p.value))
          .toList();
    }

    final brainSpots = toSpots(brainAgePoints);
    final actualSpots = toSpots(actualAgePoints);

    // === 為了處理 x 軸閏年問題的小工具們 ===
    List<int> generateYearlyTicks(DateTime start, DateTime end) {
      final ticks = <int>[];
      var tickDate = DateTime(start.year, 1, 1);
      while (tickDate.isBefore(end)) {
        ticks.add(tickDate.difference(start).inDays.toInt());
        tickDate = DateTime(tickDate.year + 1, 1, 1);
      }
      return ticks;
    }

    final customTicks = generateYearlyTicks(baseDate, endDate);
    final Map<int, Widget> tickLabelMap = {
      for (var tick in customTicks)
        tick: Text(
          "${baseDate.add(Duration(days: tick)).year}",
          style: const TextStyle(fontSize: 10),
        ),
    };
    int? findNearestKey(
      double target,
      Iterable<int> keys, {
      int threshold = 2,
    }) {
      if (keys.isEmpty) return null;

      int closest = keys.first;
      double minDiff = (closest - target).abs();

      for (final k in keys.skip(1)) {
        final diff = (k - target).abs();
        if (diff < minDiff) {
          minDiff = diff;
          closest = k;
        }
      }

      // 避免太遠的值誤判為接近
      return minDiff <= threshold ? closest : null;
    }
    // === 為了處理 x 軸閏年問題的小工具們 (結尾) ===

    return CustomCard(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 15, vertical: 10),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Text(
                  '腦齡變化趨勢',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.w500),
                ),
                SizedBox(width: 30),
                Text(
                  '● 實際年齡',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                    color: Colors.greenAccent,
                  ),
                ),
                SizedBox(width: 30),
                Text(
                  '● 腦部年齡',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                    color: Colors.orange,
                  ),
                ),
              ],
            ),

            const SizedBox(height: 20),
            AspectRatio(
              aspectRatio: 16 / 6,
              child: LineChart(
                LineChartData(
                  minX: 0,
                  maxX: maxOffset,
                  minY: minY,
                  maxY: maxY,
                  lineTouchData: LineTouchData(
                    enabled: true,
                    touchTooltipData: LineTouchTooltipData(
                      //tooltipBgColor: Colors.black54,
                      getTooltipItems: (spots) {
                        return spots.map((spot) {
                          final shownDate = baseDate.add(
                            Duration(days: spot.x.toInt()),
                          );
                          final isActual = spot.barIndex == 0;
                          final color =
                              isActual ? Colors.greenAccent : Colors.orange;
                          final text = isActual ? "實際年齡" : "腦部年齡";
                          return LineTooltipItem(
                            "$text : ${spot.y.toStringAsFixed(0)} 歲",
                            TextStyle(color: color),
                          );
                        }).toList();
                      },
                    ),
                  ),
                  gridData: FlGridData(
                    show: true,
                    drawVerticalLine: false,
                    drawHorizontalLine: true,
                    horizontalInterval: 5,
                    getDrawingHorizontalLine:
                        (value) => FlLine(
                          color: Colors.grey.withOpacity(0.3),
                          strokeWidth: 1,
                        ),
                  ),
                  titlesData: FlTitlesData(
                    topTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false),
                    ),
                    rightTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false),
                    ),
                    leftTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 32,
                        interval: 5,
                        getTitlesWidget:
                            (value, meta) => SideTitleWidget(
                              meta: meta,
                              child: Text(
                                value.toStringAsFixed(0),
                                style: const TextStyle(fontSize: 10),
                              ),
                            ),
                      ),
                    ),
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 40,
                        interval: 365.25 * intervalWeight,
                        getTitlesWidget: (value, meta) {
                          final nearest = findNearestKey(
                            value,
                            tickLabelMap.keys,
                            threshold: 10,
                          );
                          if (nearest == null) return const SizedBox.shrink();

                          final widget = tickLabelMap[nearest];
                          return SideTitleWidget(meta: meta, child: widget!);
                        },
                      ),
                    ),
                  ),
                  borderData: FlBorderData(
                    show: true,
                    border: const Border(
                      left: BorderSide(color: Colors.grey),
                      bottom: BorderSide(color: Colors.grey),
                      right: BorderSide(color: Colors.transparent),
                      top: BorderSide(color: Colors.transparent),
                    ),
                  ),
                  lineBarsData: [
                    LineChartBarData(
                      spots: actualSpots,
                      isCurved: false,
                      color: Colors.greenAccent,
                      barWidth: 1.5, // 🔹 線條變細
                      dotData: const FlDotData(show: true), // 🔹 顯示每個點
                    ),
                    LineChartBarData(
                      spots: brainSpots,
                      isCurved: false,
                      color: Colors.orange,
                      barWidth: 1.5,
                      dotData: const FlDotData(show: true),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
