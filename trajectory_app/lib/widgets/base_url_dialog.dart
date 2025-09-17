import 'package:flutter/material.dart';
import 'package:trajectory_app/services/api_service.dart';
import 'package:trajectory_app/services/auth_service.dart';

class BaseUrlDialog extends StatefulWidget {
  const BaseUrlDialog({super.key});

  static Future<void> show(BuildContext context) async {
    showDialog(context: context, builder: (context) => const BaseUrlDialog());
  }

  @override
  State<BaseUrlDialog> createState() => _BaseUrlDialogState();
}

class _BaseUrlDialogState extends State<BaseUrlDialog> {
  final TextEditingController _urlController = TextEditingController();

  @override
  void initState() {
    super.initState();
    // 一打開Dialog就帶入目前baseUrl，並把游標放到最後
    _urlController.text = ApiService.getBaseUrl();
    _urlController.selection = TextSelection.fromPosition(
      TextPosition(offset: _urlController.text.length),
    );
  }

  void _applyBaseUrl() {
    final newUrl = _urlController.text.trim();
    if (newUrl.isNotEmpty) {
      ApiService.setBaseUrl(newUrl);
      AuthService.setBaseUrl(newUrl);
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('✅ BaseUrl已切換為：$newUrl')));
      Navigator.pop(context);
    }
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('變更後端位址'),
      content: TextField(
        controller: _urlController,
        decoration: const InputDecoration(hintText: '請輸入新的 baseUrl'),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('取消'),
        ),
        ElevatedButton(onPressed: _applyBaseUrl, child: const Text('確認切換')),
      ],
    );
  }
}
