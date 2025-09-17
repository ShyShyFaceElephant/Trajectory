import 'package:flutter/material.dart';
import 'package:trajectory_app/const/constant.dart';
import 'package:trajectory_app/services/api_service.dart';
import 'package:trajectory_app/widgets/profile_widget.dart';
import 'package:trajectory_app/widgets/records_widget.dart';
import 'package:trajectory_app/widgets/reports_widget.dart';
import 'package:trajectory_app/widgets/side_menu_widget.dart';

class MemberScreen extends StatefulWidget {
  final String memberId;
  const MemberScreen({super.key, required this.memberId});

  @override
  State<MemberScreen> createState() => _MemberScreenState();
}

class _MemberScreenState extends State<MemberScreen> {
  int _selectedIndex = 0;
  late final List<Widget> mainWidgetList;
  ProfileWidget profileWidget = const ProfileWidget(
    type: 'member',
    usingLocalImage: false,
  );
  void onMenuTap(int index) {
    setState(() {
      if (index == 3) {
        Navigator.pop(context);
        return;
      }
      if (index != 2) _selectedIndex = index;
    });
  }

  void loadMemberInfo() async {
    final memberModel = await ApiService.getMemberInfo(widget.memberId);
    setState(() {
      // **這裡要重新建立 profileWidgetList，確保 UI 會更新**
      profileWidget = ProfileWidget(
        type: 'member',
        member: memberModel,
        usingLocalImage: false,
      );
    });
  }

  @override
  void initState() {
    super.initState();
    loadMemberInfo(); // 在 initState() 內執行，只執行一次
    mainWidgetList = <Widget>[
      ReportsWidget(memberId: widget.memberId),
      RecordsWidget(memberId: widget.memberId),
    ];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: _appBar(),

      body: SafeArea(
        child: _buildPage(
          SideMenuWidget(
            type: 'member',
            selectedIndex: _selectedIndex,
            onMenuTap: onMenuTap,
          ),
          mainWidgetList[_selectedIndex],
          profileWidget,
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
