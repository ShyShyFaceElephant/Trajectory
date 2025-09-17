import 'package:flutter/material.dart';
import 'package:trajectory_app/cards/brain_veiwer_card.dart';
import 'package:trajectory_app/cards/selet_record_card.dart';
import 'package:trajectory_app/models/record_model.dart';
import 'package:trajectory_app/services/api_service.dart';

class RecordsWidget extends StatefulWidget {
  final String memberId;
  const RecordsWidget({super.key, required this.memberId});

  @override
  State<RecordsWidget> createState() => _RecordsWidgetState();
}

class _RecordsWidgetState extends State<RecordsWidget> {
  List<RecordModel>? records;
  var viewerIndexList = <int>[];
  void _loadRecordList() async {
    final response = await ApiService.getMemberRecordList(widget.memberId);
    setState(() {
      records = response;
    });
  }

  void _buildBrainViewer(int index) {
    setState(() {
      if (!viewerIndexList.contains(index)) viewerIndexList.add(index);
    });
  }

  void _popBrainViewer(int index) {
    setState(() {
      if (viewerIndexList.contains(index)) viewerIndexList.remove(index);
    });
  }

  @override
  void initState() {
    super.initState();
    _loadRecordList();
  }

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.topCenter, // 設定內容向上對齊,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 20),
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            children: [
              SelectRecordCard(
                buildBrainViewer: _buildBrainViewer,
                data: records,
              ),
              const SizedBox(height: 18),
              ListView.builder(
                shrinkWrap: true, // ✅ 讓 ListView 只佔用實際內容的高度
                physics:
                    const NeverScrollableScrollPhysics(), // ✅ 禁止內部滾動，避免與外部滾動衝突
                reverse: true,
                itemCount: viewerIndexList.length,
                itemBuilder: (context, index) {
                  var record = records![viewerIndexList[index]];
                  return Column(
                    children: [
                      BrainViewerCard(
                        memberId: widget.memberId,
                        record: record,
                        recordIndex: viewerIndexList[index],
                        popBrainViewer: _popBrainViewer,
                      ),
                      const SizedBox(height: 18),
                    ],
                  );
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}
