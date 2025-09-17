import 'package:flutter/material.dart';
import 'package:trajectory_app/const/constant.dart';
import 'package:trajectory_app/services/auth_service.dart';
import 'package:trajectory_app/widgets/base_url_dialog.dart';

class SigninScreen extends StatelessWidget {
  const SigninScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(color: cardBackgroundColor),
      child: Center(
        child: SizedBox(
          width: 400,
          child: DefaultTabController(
            length: 2,
            child: Scaffold(
              backgroundColor: Colors.transparent,
              body: Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: 32.0,
                  vertical: 100.0,
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    logoSetBaseUrl(context),
                    const SizedBox(height: 20),
                    const TabBar(
                      indicatorColor: Colors.tealAccent,
                      labelColor: Colors.white,
                      unselectedLabelColor: Colors.grey,
                      tabs: [Tab(text: '一般登入'), Tab(text: '醫師登入')],
                    ),
                    const Expanded(
                      child: TabBarView(
                        children: [
                          SigninForm(role: 'member', route: '/memberScreen'),
                          SigninForm(role: 'manager', route: '/managerScreen'),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  GestureDetector logoSetBaseUrl(BuildContext context) {
    return GestureDetector(
      onLongPress: () {
        BaseUrlDialog.show(context);
      },
      child: const Image(
        image: AssetImage('assets/images/Trajectory-white.png'),
        height: 70,
        fit: BoxFit.cover,
      ),
    );
  }
}

class SigninForm extends StatefulWidget {
  final String role;
  final String route;
  const SigninForm({super.key, required this.role, required this.route});
  @override
  State<SigninForm> createState() => _SigninFormState();
}

class _SigninFormState extends State<SigninForm> {
  final _usernameController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _isLoading = false;
  String? _errorMessage;

  Future<void> _handleSignin() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    final username = _usernameController.text.trim();
    final password = _passwordController.text.trim();

    if (username.isEmpty || password.isEmpty) {
      setState(() {
        _isLoading = false;
        _errorMessage = '帳號與密碼不得為空';
      });
      return;
    }

    try {
      final response = await AuthService.signin(
        username,
        password,
        widget.role,
      );
      if (response == true) {
        /******** 登入成功 *********/
        widget.role == 'manager'
            ? Navigator.pushNamed(context, '/managerScreen')
            : Navigator.pushNamed(
              context,
              '/memberScreen',
              arguments: _usernameController.text.trim(), // 傳遞 memberId
            );
      } else {
        setState(() {
          _errorMessage = "帳號或密碼錯誤";
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = "網路連線錯誤";
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 24.0),
      child: Column(
        children: [
          TextField(
            controller: _usernameController,
            cursorColor: Colors.tealAccent, // 修改游標顏色
            decoration: InputDecoration(
              hintText: '帳號',
              hintStyle: const TextStyle(color: Colors.grey),
              filled: true,
              fillColor: cardBackgroundColor,
              border: OutlineInputBorder(
                borderSide: const BorderSide(
                  color: Colors.tealAccent,
                  width: 2.0,
                ), // 修改點選時的邊框顏色
                borderRadius: BorderRadius.circular(8.0),
              ),
            ),
            style: const TextStyle(color: Colors.white),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _passwordController,
            cursorColor: Colors.tealAccent, // 修改游標顏色
            obscureText: true,
            decoration: InputDecoration(
              hintText: '密碼',
              hintStyle: const TextStyle(color: Colors.grey),
              filled: true,
              fillColor: cardBackgroundColor,
              border: OutlineInputBorder(
                borderSide: const BorderSide(
                  color: Colors.tealAccent,
                  width: 2.0,
                ), // 修改點選時的邊框顏色
                borderRadius: BorderRadius.circular(8.0),
              ),
            ),
            style: const TextStyle(color: Colors.white),
          ),
          const SizedBox(height: 16),
          /* 錯誤訊息檢查 */
          if (_errorMessage != null)
            Text(
              _errorMessage!,
              style: const TextStyle(color: Colors.red, fontSize: 14),
            ),
          const SizedBox(height: 16),
          /* 錯誤訊息檢查 */
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: selectionColor,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8.0),
                ),
              ),
              onPressed: _isLoading ? null : () => _handleSignin(),
              child:
                  _isLoading
                      ? const CircularProgressIndicator(color: Colors.white)
                      : const Text(
                        '登入',
                        style: TextStyle(fontSize: 18, color: Colors.white),
                      ),
            ),
          ),
        ],
      ),
    );
  }
}
