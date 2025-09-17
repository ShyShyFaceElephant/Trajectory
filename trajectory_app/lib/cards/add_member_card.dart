import 'package:flutter/material.dart';
import 'package:trajectory_app/cards/custom_card.dart';
import 'package:trajectory_app/models/member_model.dart';
import 'package:trajectory_app/services/api_service.dart';
import 'package:file_picker/file_picker.dart'; // 引入 file_picker
import 'dart:io'; // 用於 File 類型

class AddMemberCard extends StatefulWidget {
  final VoidCallback? loadManagerInfo; // 為了更新醫生的成員數
  final void Function(MemberModel memberModel, File? localImage)?
  loaAddMemberMemberPreview; // 為了預覽欲新增之成員
  const AddMemberCard({
    super.key,
    this.loadManagerInfo,
    this.loaAddMemberMemberPreview,
  });

  @override
  State<AddMemberCard> createState() => _AddMemberCardState();
}

class _AddMemberCardState extends State<AddMemberCard> {
  final ValueNotifier<File?> _imageNotifier = ValueNotifier<File?>(
    null,
  ); // 監聽是否有選擇檔案
  final _nameController = TextEditingController();
  final _idController = TextEditingController();
  final _dateController = TextEditingController();
  final _sexController = TextEditingController();
  final _nameFocusNode = FocusNode();
  final _idFocusNode = FocusNode();
  final _dateFocusNode = FocusNode();
  final _sexFocusNode = FocusNode();
  File? _selectedImage; // 用於儲存選擇的圖片
  String yyyyPreview = '--',
      mmPreview = '--',
      ddPreview = '--',
      namePreview = '--',
      idPreview = '--',
      sexPreview = '--';
  @override
  void dispose() {
    _nameController.dispose();
    _idController.dispose();
    _dateController.dispose();
    _sexController.dispose();
    _nameFocusNode.dispose();
    _idFocusNode.dispose();
    _dateFocusNode.dispose();
    _sexFocusNode.dispose();
    super.dispose();
  }

  /********************************* 選擇圖片 ****************************/

  Future<void> _pickImage() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      allowMultiple: false,
    );
    if (result != null && result.files.single.path != null) {
      final image = File(result.files.single.path!);
      setState(() {
        _selectedImage = image;
      });
      _imageNotifier.value = image; // ✅ 同步更新 Notifier
    }
  }

  /********************************* 送出表單 ****************************/
  Future<void> _submitForm() async {
    if (_nameController.text.isEmpty ||
        _idController.text.isEmpty ||
        _dateController.text.isEmpty ||
        _sexController.text.isEmpty) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('請填寫所有必填欄位')));
      return;
    }
    final memberModel = buildCurrentMember();
    // 調用 API，傳入 member 和圖片
    final success = await ApiService.memberSignup(
      memberModel,
      _selectedImage!,
    ); // 使用 ! 斷言非空
    if (success) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('註冊成功！')));
      // 清空表單
      _nameController.clear();
      _idController.clear();
      _dateController.clear();
      _sexController.clear();
      setState(() {
        _selectedImage = null; // 清空圖片
      });
      // 更新醫生成員數量
      widget.loadManagerInfo!();
    } else {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('註冊失敗，請稍後再試')));
    }
  }

  MemberModel buildCurrentMember() {
    final memberModel = MemberModel(
      id: idPreview,
      name: namePreview,
      sex: sexPreview,
      yyyy: yyyyPreview,
      mm: mmPreview,
      dd: ddPreview,
    );
    return memberModel;
  }

  /********************************* 送出表單 ****************************/
  @override
  void initState() {
    super.initState();
    _nameFocusNode.addListener(() {
      if (!_nameFocusNode.hasFocus) {
        print("姓名失焦：${_nameController.text}");
        String text = _nameController.text.trimRight();
        _nameController.text = text;
        namePreview = text == '' ? '--' : text;
        widget.loaAddMemberMemberPreview!(buildCurrentMember(), _selectedImage);
      }
    });

    _idFocusNode.addListener(() {
      if (!_idFocusNode.hasFocus) {
        print("身份證字號失焦：${_idController.text}");
        String text = _idController.text.trimRight();
        _idController.text = text;
        idPreview = text == '' ? '--' : text;
        widget.loaAddMemberMemberPreview!(buildCurrentMember(), _selectedImage);
      }
    });

    _dateFocusNode.addListener(() {
      if (!_dateFocusNode.hasFocus) {
        print("出生日期失焦：${_dateController.text}");
        // 解析日期
        String yyyy = '--';
        String mm = '--';
        String dd = '--';
        if (_dateController.text.isNotEmpty) {
          final dateParts = _dateController.text.split('/');
          if (dateParts.length == 3) {
            yyyy = dateParts[0];
            mm = dateParts[1];
            dd = dateParts[2];
          } else {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('日期格式錯誤，請使用 YYYY/MM/DD')),
            );
            return;
          }
        }
        yyyyPreview = yyyy;
        mmPreview = mm;
        ddPreview = dd;
        widget.loaAddMemberMemberPreview!(buildCurrentMember(), _selectedImage);
      }
    });

    _sexFocusNode.addListener(() {
      if (!_sexFocusNode.hasFocus) {
        print("性別失焦：${_sexController.text}");
        String text = _sexController.text.trimRight();
        _sexController.text = text;
        sexPreview = text == '' ? '--' : text;
        widget.loaAddMemberMemberPreview!(buildCurrentMember(), _selectedImage);
      }
    });

    _imageNotifier.addListener(() {
      if (_imageNotifier.value != null) {
        print("✅ 監聽到圖片被選擇: ${_imageNotifier.value!.path}");
        widget.loaAddMemberMemberPreview!(buildCurrentMember(), _selectedImage);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return CustomCard(
      width: 425,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 50, vertical: 10),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '新增成員',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 16),
            _buildInputRow(
              '姓名',
              '姓名',
              true,
              controller: _nameController,
              focusNode: _nameFocusNode,
            ),
            _buildInputRow(
              '身份證字號',
              '身份證字號',
              true,
              controller: _idController,
              focusNode: _idFocusNode,
            ),
            _buildInputRow(
              '出生日期',
              'YYYY/MM/DD',
              true,
              controller: _dateController,
              focusNode: _dateFocusNode,
            ),
            _buildInputRow(
              '性別',
              'M/F',
              true,
              controller: _sexController,
              focusNode: _sexFocusNode,
            ),
            _buildImagePickerRow(),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(flex: 3, child: Container()),
                const SizedBox(width: 20),
                Expanded(
                  flex: 2,
                  child: OutlinedButton(
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(color: Colors.white),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8.0),
                      ),
                      padding: const EdgeInsets.symmetric(
                        horizontal: 20,
                        vertical: 12,
                      ),
                    ),
                    onPressed: _submitForm, // 送出按鈕
                    child: const Text(
                      '送出',
                      style: TextStyle(color: Colors.white),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildImagePickerRow() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          const Text(
            '上傳照片',
            style: TextStyle(color: Colors.white70, fontSize: 16),
          ),
          SizedBox(
            height: 40,
            width: 200,
            child: ValueListenableBuilder<File?>(
              valueListenable: _imageNotifier,
              builder: (context, selectedImage, _) {
                return OutlinedButton(
                  onPressed: _pickImage,
                  style: OutlinedButton.styleFrom(
                    side: const BorderSide(color: Colors.white),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8.0),
                    ),
                    padding: const EdgeInsets.all(0),
                  ),
                  child: Center(
                    child:
                        selectedImage == null
                            ? const Text(
                              '點擊選擇照片',
                              style: TextStyle(color: Colors.white54),
                            )
                            : Image.file(
                              selectedImage,
                              fit: BoxFit.cover,
                              height: 40,
                              width: 200,
                            ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

Widget _buildInputRow(
  String label,
  String hint,
  bool enabled, {
  TextEditingController? controller,
  FocusNode? focusNode,
}) {
  return Padding(
    padding: const EdgeInsets.symmetric(vertical: 8.0),
    child: Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: const TextStyle(color: Colors.white70, fontSize: 16),
        ),
        SizedBox(
          height: 40,
          width: 200,
          child: TextField(
            focusNode: focusNode,
            controller: controller,
            enabled: enabled,
            style: const TextStyle(color: Colors.white),
            decoration: InputDecoration(
              filled: true,
              fillColor:
                  enabled
                      ? Colors.transparent
                      : const Color.fromARGB(255, 55, 56, 74),
              hintText: hint,
              hintStyle: const TextStyle(color: Colors.white54),
              border: OutlineInputBorder(
                borderSide: const BorderSide(color: Colors.red),
                borderRadius: BorderRadius.circular(8.0),
              ),
              focusedBorder: OutlineInputBorder(
                borderSide: const BorderSide(color: Colors.white),
                borderRadius: BorderRadius.circular(8.0),
              ),
              contentPadding: const EdgeInsets.symmetric(
                horizontal: 12,
                vertical: 10,
              ),
            ),
          ),
        ),
      ],
    ),
  );
}
