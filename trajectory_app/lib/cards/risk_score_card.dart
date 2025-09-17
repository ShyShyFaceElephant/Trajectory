import 'package:flutter/material.dart';
import 'package:trajectory_app/cards/custom_card.dart';

class RiskScoreCard extends StatefulWidget {
  final String? riskScore;
  const RiskScoreCard({super.key, required this.riskScore});

  @override
  State<RiskScoreCard> createState() => _RiskScoreCard();
}

class _RiskScoreCard extends State<RiskScoreCard> {
  @override
  Widget build(BuildContext context) {
    return CustomCard(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Text(
                  '阿茲海默症評估',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                SizedBox(width: 10),
                Icon(Icons.info_outline_rounded, color: Colors.grey, size: 16),
                SizedBox(width: 8),
                Text(
                  "AI 僅供輔助判斷，請依專業醫師臨床評估為準。",
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey,
                    height: 1.4,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            if (widget.riskScore == null)
              const Center(
                child: const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                  child: Text(
                    "未參與過認知測驗，無法提供 AI 評估結果。",
                    style: TextStyle(fontSize: 14, color: Colors.white70),
                  ),
                ),
              )
            else
              _riskResult(),
          ],
        ),
      ),
    );
  }

  Widget _riskResult() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 30),
      child: Stack(
        alignment: Alignment.topCenter,
        clipBehavior: Clip.none,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _riskSegment(
                Colors.green,
                '認知正常',
                widget.riskScore == 'CN(正常認知)',
              ),
              _riskSegment(
                Colors.yellow,
                '輕度認知障礙',
                widget.riskScore == 'MCI(輕度認知障礙)',
              ),
              _riskSegment(
                Colors.red,
                '阿茲海默症',
                widget.riskScore == 'AD(阿茲海默症)',
              ),
            ],
          ),
        ],
      ),
    );
  }
}

Widget _riskSegment(Color color, String label, bool flag) {
  final displayColor = flag ? color : color.withOpacity(0.2);

  return Expanded(
    child: Column(
      children: [
        const SizedBox(height: 20),
        Container(height: 16, color: displayColor),
        const SizedBox(height: 4),
        Text(
          label,
          style: TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.bold,
            color: displayColor,
          ),
        ),
      ],
    ),
  );
}
