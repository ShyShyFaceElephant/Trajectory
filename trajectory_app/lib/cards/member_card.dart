import 'package:flutter/material.dart';
import 'package:trajectory_app/cards/custom_card.dart';
import 'package:trajectory_app/const/constant.dart';
import 'package:trajectory_app/models/member_model.dart';
import 'package:trajectory_app/services/api_service.dart';

class MemberCard extends StatelessWidget {
  final int index;
  final List<MemberModel> memberList;
  const MemberCard({super.key, required this.index, required this.memberList});
  @override
  Widget build(BuildContext context) {
    return CustomCard(
      width: 500,
      color: cardBackgroundColor, // 背景色
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
        child: Row(
          children: [
            // 頭像
            Container(
              width: 130, // 直徑 = 2 * radius
              height: 130,
              padding: const EdgeInsets.all(5),
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(
                  color: selectionColor, // 邊框顏色
                  width: 2, // 邊框寬度
                ),
              ),
              /*------------個人照顯示 (非同步建構)---------------*/
              child: FutureBuilder<String?>(
                future: ApiService.getMemberImage(memberList[index].id),
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.waiting) {
                    return const Center(
                      child: CircularProgressIndicator(color: Colors.white),
                    );
                  } else {
                    final imageUrl = snapshot.data;
                    return CircleAvatar(
                      radius: 64,
                      backgroundColor: Colors.transparent,
                      backgroundImage:
                          (imageUrl != null)
                              ? NetworkImage(imageUrl)
                              : const AssetImage('/assets/images/avatar.png')
                                  as ImageProvider,
                    );
                  }
                },
              ),
              /*------------個人照顯示 結束---------------*/
            ),
            const SizedBox(width: 16),
            // 文字資訊
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 20),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          memberList[index].name,
                          style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),

                        ElevatedButton(
                          style: ElevatedButton.styleFrom(
                            backgroundColor: selectionColor,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(5.0),
                            ),
                            padding: const EdgeInsets.symmetric(
                              vertical: 1.0,
                              horizontal: 16.0,
                            ),
                          ),
                          onPressed: () {
                            Navigator.pushNamed(
                              context,
                              '/memberScreen',
                              arguments: memberList[index].id,
                            );
                          },
                          child: const Text(
                            '選擇',
                            style: TextStyle(color: Colors.white, fontSize: 16),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const Divider(
                    // 分隔線
                    color: selectionColor, // 直線顏色
                    thickness: 1.5, // 直線粗細
                  ),
                  _buildInfoRow('身份證字號', memberList[index].id),
                  _buildInfoRow(
                    '出生年月日',
                    '${memberList[index].yyyy}/${memberList[index].mm}/${memberList[index].dd}',
                  ),
                  _buildInfoRow('性別', memberList[index].sex),
                  _buildInfoRow('影像紀錄', memberList[index].numRecords),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 1, horizontal: 16.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: const TextStyle(
              color: Colors.white54,
              fontWeight: FontWeight.w500,
              fontSize: 16,
            ),
          ),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white70,
              fontWeight: FontWeight.w500,
              fontSize: 16,
            ),
          ),
        ],
      ),
    );
  }
}
