import 'package:flutter/material.dart';
import 'package:trajectory_app/models/menu_model.dart';

class SideMenuData {
  final userMenu = const <MenuModel>[
    MenuModel(icon: Icons.person, title: '腦部分析'),
    MenuModel(icon: Icons.history, title: '影像紀錄'),
    MenuModel(icon: Icons.run_circle, title: '健康指南'),
    MenuModel(icon: Icons.logout, title: '登出'),
  ];
  final managerMenu = const <MenuModel>[
    MenuModel(icon: Icons.manage_accounts, title: '成員管理'),
    MenuModel(icon: Icons.upload_file, title: '上傳影像'),
    MenuModel(icon: Icons.person_add_alt_1, title: '新增成員'),
    MenuModel(icon: Icons.logout, title: '登出'),
  ];
}
