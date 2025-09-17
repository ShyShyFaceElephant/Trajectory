import 'dart:io';

import 'package:flutter/material.dart';
import 'package:trajectory_app/const/constant.dart';
import 'package:trajectory_app/models/manager_model.dart';
import 'package:trajectory_app/models/member_model.dart';
import 'package:trajectory_app/services/api_service.dart';

class ProfileWidget extends StatelessWidget {
  final String type;
  final bool usingLocalImage;
  final File? localImage;
  final MemberModel member;
  final ManagerModel manager;

  const ProfileWidget({
    super.key,
    required this.type,
    required this.usingLocalImage,
    this.member = const MemberModel(),
    this.manager = const ManagerModel(),
    this.localImage,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.only(left: 15, right: 15, top: 50),
      color: backgroundColor, // 深色背景
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          const SizedBox(height: 20),
          const Text(
            '個人檔案',
            style: TextStyle(
              color: Colors.white,
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 20),
          usingLocalImage == false
              ? (type ==
                      'manager' // 沒有本地路徑的話就網路上找
                  ? _managerProfileImage(manager)
                  : _memberProfileImage(member))
              : _localProfileImage(member, localImage), // 有本地路徑就從本地抓圖片
          const SizedBox(height: 20),
          type == 'manager' ? _managerInfo(manager) : _memberInfo(member),
        ],
      ),
    );
  }
}

Widget _localProfileImage(MemberModel member, File? localImage) {
  return Container(
    width: 180,
    height: 180,
    padding: const EdgeInsets.all(5),
    decoration: BoxDecoration(
      shape: BoxShape.circle,
      border: Border.all(color: selectionColor, width: 2),
    ),
    child: CircleAvatar(
      radius: 64,
      backgroundColor: Colors.transparent,
      backgroundImage: localImage != null ? FileImage(localImage) : null,
    ),
  );
}

Widget _managerProfileImage(ManagerModel manager) {
  return FutureBuilder<String?>(
    future: ApiService.getManagerImage(manager.id),
    builder: (context, snapshot) {
      final imageUrl = snapshot.data;

      return Container(
        width: 180,
        height: 180,
        padding: const EdgeInsets.all(5),
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(
            color: const Color.fromARGB(255, 250, 250, 152),
            width: 2,
          ),
        ),
        child: CircleAvatar(
          radius: 64,
          backgroundColor: Colors.transparent,
          backgroundImage:
              (snapshot.connectionState == ConnectionState.done &&
                      imageUrl != null)
                  ? NetworkImage(imageUrl)
                  : null,
        ),
      );
    },
  );
}

Widget _memberProfileImage(MemberModel member) {
  return FutureBuilder<String?>(
    future: ApiService.getMemberImage(member.id),
    builder: (context, snapshot) {
      final imageUrl = snapshot.data;

      return Container(
        width: 180,
        height: 180,
        padding: const EdgeInsets.all(5),
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(color: selectionColor, width: 2),
        ),
        child: CircleAvatar(
          radius: 64,
          backgroundColor: Colors.transparent,
          backgroundImage:
              (snapshot.connectionState == ConnectionState.done &&
                      imageUrl != null)
                  ? NetworkImage(imageUrl)
                  : null,
        ),
      );
    },
  );
}

Column _memberInfo(MemberModel member) {
  return Column(
    children: [
      Text(
        member.name,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 20,
          fontWeight: FontWeight.bold,
        ),
      ),
      const Divider(
        // 分隔線
        color: selectionColor, // 直線顏色
        thickness: 1.0, // 直線粗細
        indent: 10.0, // 左側縮進
        endIndent: 10.0, // 右側縮進
      ),
      _buildInfoRow('身分證字號', member.id),
      _buildInfoRow('出生日期', '${member.yyyy}/${member.mm}/${member.dd}'),
      _buildInfoRow('性別', member.sex),
      _buildInfoRow('影像紀錄', member.numRecords),
      const Divider(
        color: selectionColor, // 直線顏色
        thickness: 1.0, // 直線粗細
        indent: 10.0, // 左側縮進
        endIndent: 10.0, // 右側縮進
      ),
    ],
  );
}

Column _managerInfo(ManagerModel manager) {
  return Column(
    children: [
      Text(
        manager.name,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 20,
          fontWeight: FontWeight.bold,
        ),
      ),
      const Divider(
        // 分隔線
        color: selectionColor, // 直線顏色
        thickness: 1.0, // 直線粗細
        indent: 10.0, // 左側縮進
        endIndent: 10.0, // 右側縮進
      ),
      _buildInfoRow('編號', manager.id),
      _buildInfoRow('科別', manager.department),
      _buildInfoRow('成員人數', manager.numMembers),
      const Divider(
        color: selectionColor, // 直線顏色
        thickness: 1.0, // 直線粗細
        indent: 10.0, // 左側縮進
        endIndent: 10.0, // 右側縮進
      ),
    ],
  );
}

Widget _buildInfoRow(String label, String value) {
  return Padding(
    padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 16.0),
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
