import 'package:flutter/material.dart';
import 'package:trajectory_app/cards/member_card.dart';
import 'package:trajectory_app/models/member_model.dart';
import 'package:trajectory_app/services/api_service.dart';

class MemberListWidget extends StatefulWidget {
  const MemberListWidget({super.key});

  @override
  State<MemberListWidget> createState() => _MemberListWidgetState();
}

class _MemberListWidgetState extends State<MemberListWidget> {
  late List<MemberModel> data; // 明確指定型別為 List<MemberModel>
  bool _isLoading = true; // 用於顯示載入狀態

  @override
  void initState() {
    _fetchData(); // 在初始化時載入資料
    super.initState();
  }

  Future<void> _fetchData() async {
    try {
      data = await ApiService.getMemberList();
      setState(() {
        _isLoading = false; // 資料載入完成
      });
    } catch (e) {
      print("Error fetching data: $e");
      setState(() {
        _isLoading = false;
        data = []; // 失敗時設為空列表
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) return const Center(child: CircularProgressIndicator());

    return Align(
      alignment: Alignment.topCenter, // 設定內容向上對齊
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 20),
        child: ListView.builder(
          shrinkWrap: true, // 限制 ListView 的大小
          itemCount: data.length,
          itemBuilder:
              (context, index) => Column(
                children: [
                  MemberCard(memberList: data, index: index),
                  const SizedBox(height: 10), // 這裡控制間距，例如 10
                ],
              ),
        ),
      ),
    );
  }
}
