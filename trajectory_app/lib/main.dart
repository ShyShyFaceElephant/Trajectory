import 'package:flutter/material.dart';
import 'package:trajectory_app/screens/member_screen.dart';
import 'package:trajectory_app/screens/manager_screen.dart';
import 'package:trajectory_app/screens/signin_screen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Trajectory',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: Colors.black,
        brightness: Brightness.dark,
      ),
      home: const SigninScreen(),
      onGenerateRoute: (settings) {
        // 根據路由名稱生成對應的畫面
        switch (settings.name) {
          case '/managerScreen':
            return MaterialPageRoute(
              builder: (context) => const ManagerScreen(),
            );
          case '/memberScreen':
            // 從 settings.arguments 提取 memberId
            final String? memberId = settings.arguments as String?;
            if (memberId == null) {
              // 如果缺少 memberId，可以導向錯誤畫面或拋出異常
              return MaterialPageRoute(
                builder:
                    (context) => const Scaffold(
                      body: Center(child: Text('缺少 memberId')),
                    ),
              );
            }
            return MaterialPageRoute(
              builder: (context) => MemberScreen(memberId: memberId),
            );
          default:
            // 未知路由，顯示錯誤畫面
            return MaterialPageRoute(
              builder:
                  (context) =>
                      const Scaffold(body: Center(child: Text('路由不存在'))),
            );
        }
      },
    );
  }
}
