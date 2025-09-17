import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:trajectory_app/const/constant.dart'; // 儲存中小型檔案之套件

class AuthService {
  static String baseUrl = backendUrl; // 你的後端 API 位址
  static void setBaseUrl(String newUrl) {
    baseUrl = newUrl;
  }

  static String getBaseUrl() {
    return baseUrl; // 取目前用的
  }

  // 登入方法
  static Future<bool> signin(
    String username,
    String password,
    String role,
  ) async {
    final url =
        role == 'manager'
            ? Uri.parse('$baseUrl/manager/Signin')
            : Uri.parse('$baseUrl/member/Signin');
    final response = await http.post(
      url,
      body: jsonEncode({"id": username, "password": password}),
      headers: {"Content-Type": "application/json"},
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      String token =
          role == 'manager' ? data["manager_token"] : data["member_token"];
      // 儲存 token
      SharedPreferences prefs = await SharedPreferences.getInstance();
      await prefs.setString('token', token);
      return true;
    }
    return false;
  }

  // 取得已儲存的 Token
  static Future<String?> getToken() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    return prefs.getString('token');
  }

  // 登出
  static Future<void> logout() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    await prefs.remove('token');
  }
}
