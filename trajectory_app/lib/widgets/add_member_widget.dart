import 'dart:io';

import 'package:flutter/material.dart';
import 'package:trajectory_app/cards/add_member_card.dart';
import 'package:trajectory_app/models/member_model.dart';

class AddMemberWidget extends StatefulWidget {
  final VoidCallback? loadManagerInfo; // 為了更新醫生的成員數
  final void Function(MemberModel memberModel, File? localImage)?
  loaAddMemberMemberPreview; // 為了預覽欲新增之成員
  const AddMemberWidget({
    super.key,
    this.loadManagerInfo,
    this.loaAddMemberMemberPreview,
  });

  @override
  State<AddMemberWidget> createState() => _AddMemberWidgetState();
}

class _AddMemberWidgetState extends State<AddMemberWidget> {
  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.topCenter,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 20),
        child: Column(
          children: [
            AddMemberCard(
              loadManagerInfo: widget.loadManagerInfo,
              loaAddMemberMemberPreview: widget.loaAddMemberMemberPreview,
            ),
          ],
        ),
      ),
    );
  }
}
