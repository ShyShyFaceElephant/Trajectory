import 'dart:io';

import 'package:flutter/material.dart';
import 'package:trajectory_app/const/constant.dart';
import 'package:trajectory_app/models/member_model.dart';
import 'package:trajectory_app/services/api_service.dart';
import 'package:trajectory_app/services/auth_service.dart';
import 'package:trajectory_app/widgets/add_member_widget.dart';
import 'package:trajectory_app/widgets/member_list_widget.dart';
import 'package:trajectory_app/widgets/profile_widget.dart';
import 'package:trajectory_app/widgets/side_menu_widget.dart';
import 'package:trajectory_app/widgets/upload_form_widget.dart';

class ManagerScreen extends StatefulWidget {
  const ManagerScreen({super.key});

  @override
  State<ManagerScreen> createState() => _ManagerScreenState();
}

class _ManagerScreenState extends State<ManagerScreen> {
  int _selectedIndex = 0;
  void onMenuTap(int index) {
    setState(() {
      // 登出
      if (index == 3) {
        AuthService.logout();
        Navigator.pop(context);
        return;
      }
      _selectedIndex = index;
    });
  }

  // 更新 mainWidgetList，將回調傳給 AddMemberWidget
  late final List<Widget> mainWidgetList = [
    const MemberListWidget(),
    UploadFormWidget(loadUploadFromMemberPreview: loadUploadFromMemberPreview),
    AddMemberWidget(
      loadManagerInfo: loadManagerInfo,
      loaAddMemberMemberPreview: loaAddMemberMemberPreview,
    ), // 傳遞回調
  ];
  List<Widget> profileWidgetList = [
    const ProfileWidget(type: 'manager', usingLocalImage: false), // 成員管理頁
    const ProfileWidget(type: 'member', usingLocalImage: false), // 上傳影像頁
    const ProfileWidget(type: 'member', usingLocalImage: false), // 新增成員頁
  ];

  void loadUploadFromMemberPreview(String memberId) async {
    final memberModel = await ApiService.getMemberInfo(memberId);
    if (!mounted) return; // <--- 這行是關鍵 避免頁面被刪除還呼叫初始化
    setState(() {
      // 成員管理頁
      profileWidgetList[1] = ProfileWidget(
        type: 'member',
        member: memberModel,
        usingLocalImage: false,
      );
    });
  }

  void loaAddMemberMemberPreview(MemberModel memberModel, File? localImage) {
    setState(() {
      profileWidgetList[2] = ProfileWidget(
        type: 'member',
        usingLocalImage: true,
        member: memberModel,
        localImage: localImage,
      );
    });
  }

  void loadManagerInfo() async {
    final managerModel = await ApiService.getManagerInfo();
    setState(() {
      // 成員管理頁
      profileWidgetList[0] = ProfileWidget(
        type: 'manager',
        usingLocalImage: false,
        manager: managerModel,
      );
    });
  }

  @override
  void initState() {
    super.initState();
    loadManagerInfo(); // 在 initState() 內執行，只執行一次
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: _appBar(),

      body: SafeArea(
        child: _buildPage(
          SideMenuWidget(
            type: 'manager',
            selectedIndex: _selectedIndex,
            onMenuTap: onMenuTap,
          ),
          mainWidgetList[_selectedIndex],
          profileWidgetList[_selectedIndex],
        ),
      ),
    );
  }
}

Row _buildPage(
  SideMenuWidget sideMenuWidget,
  Widget mainWidget,
  Widget profileWidget,
) {
  return Row(
    children: [
      // Expanded (彈性布局)，通常用於Row和Col，使用flex分配空間比例，flex總和為12
      Expanded(flex: 2, child: SizedBox(child: sideMenuWidget)),
      Expanded(flex: 9, child: SizedBox(child: mainWidget)),
      Expanded(flex: 3, child: SizedBox(child: profileWidget)),
    ],
  );
}

AppBar _appBar() {
  return AppBar(
    automaticallyImplyLeading: false,
    title: const Padding(
      padding: EdgeInsets.only(left: 25),
      child: Image(
        image: AssetImage('assets/images/Trajectory-white.png'),
        height: 40,
        fit: BoxFit.cover,
      ),
    ),
    backgroundColor: appBarColor,
  );
}
