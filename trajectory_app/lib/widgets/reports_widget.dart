import 'package:flutter/material.dart';
import 'package:trajectory_app/cards/line_char_card.dart';
import 'package:trajectory_app/cards/risk_score_card.dart';
import 'package:trajectory_app/models/chart_point_model.dart';
import 'package:trajectory_app/models/record_model.dart';
import 'package:trajectory_app/services/api_service.dart';

class ReportsWidget extends StatefulWidget {
  final String memberId;
  const ReportsWidget({super.key, required this.memberId});

  @override
  State<ReportsWidget> createState() => _ReportsWidgetState();
}

class _ReportsWidgetState extends State<ReportsWidget> {
  List<ChartPointModel> brainAgePoints = [];
  List<ChartPointModel> actualAgePoints = [];
  String? riskScore;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    loadPoints();
  }

  Future<void> loadPoints() async {
    try {
      final List<RecordModel> result = await ApiService.getMemberRecordList(
        widget.memberId,
      );
      final points = extractChartPoints(result);
      setState(() {
        brainAgePoints = points['brain'] ?? [];
        actualAgePoints = points['actual'] ?? [];
        for (var record in result) {
          riskScore = (record.riskScore == '--') ? riskScore : record.riskScore;
        }
        _isLoading = false;
      });
    } catch (e) {
      print("❌ 載入報告資料失敗：$e");
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Center(
        child: CircularProgressIndicator(color: Colors.white),
      );
    }
    return SingleChildScrollView(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 20),
        child: Column(
          children: [
            RiskScoreCard(riskScore: riskScore),
            const SizedBox(height: 18),
            LineChartCard(
              brainAgePoints: brainAgePoints,
              actualAgePoints: actualAgePoints,
            ),
          ],
        ),
      ),
    );
  }
}
