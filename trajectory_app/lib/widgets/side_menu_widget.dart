import 'package:flutter/material.dart';
import 'package:trajectory_app/const/constant.dart';
import 'package:trajectory_app/data/side_menu_data.dart';

class SideMenuWidget extends StatefulWidget {
  final String type;
  final Function(int) onMenuTap;
  final int selectedIndex;
  const SideMenuWidget({
    super.key,
    required this.type,
    required this.onMenuTap,
    required this.selectedIndex,
  });

  @override
  State<SideMenuWidget> createState() => _SideMenuWidgetState();
}

class _SideMenuWidgetState extends State<SideMenuWidget> {
  @override
  Widget build(BuildContext context) {
    final data = SideMenuData();
    return Container(
      color: backgroundColor,
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 80),
      child: ListView.builder(
        itemCount: data.userMenu.length,
        itemBuilder: (context, index) => buildMenuEntry(data, index),
      ),
    );
  }

  Widget buildMenuEntry(SideMenuData data, int index) {
    final isSelected = widget.selectedIndex == index;
    final menuMap = {'manager': data.managerMenu, 'member': data.userMenu};
    final menu = menuMap[widget.type] ?? data.userMenu; // 預設值
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 5),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(10),
        color: isSelected ? selectionColor : Colors.transparent,
      ),
      child: InkWell(
        //帶有水波點擊效果的按鈕
        onTap: () => widget.onMenuTap(index),
        child: Row(
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 3),
              child: Icon(
                menu[index].icon,
                color: isSelected ? Colors.black : Colors.grey,
              ),
            ),
            Text(
              menu[index].title,
              style: TextStyle(
                color: isSelected ? Colors.black : Colors.grey,
                fontWeight: FontWeight.bold,
                fontSize: 16,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
