import 'package:flutter/material.dart';
import 'package:trajectory_app/cards/upload_form_card.dart';

class UploadFormWidget extends StatelessWidget {
  final void Function(String memberId)?
  loadUploadFromMemberPreview; // 為了預覽當先選定成員
  const UploadFormWidget({
    super.key,
    required this.loadUploadFromMemberPreview,
  });

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      child: Align(
        alignment: Alignment.topCenter,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 20),
          child: Column(
            children: [
              UploadFormCard(
                loadUploadFromMemberPreview: loadUploadFromMemberPreview,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
