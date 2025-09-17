import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:trajectory_app/const/constant.dart';
import 'package:trajectory_app/models/manager_model.dart';
import 'package:trajectory_app/models/member_model.dart';
import 'package:trajectory_app/models/record_model.dart';
import 'package:trajectory_app/services/auth_service.dart'; // 儲存中小型檔案之套件
import 'package:path_provider/path_provider.dart';
import 'package:http_parser/http_parser.dart';
import 'dart:io'; // 用於 File 類型

class ApiService {
  static String _baseUrl = backendUrl; // 你的後端 API 位址
  static void setBaseUrl(String newUrl) {
    _baseUrl = newUrl;
  }

  static String getBaseUrl() {
    return _baseUrl; // 取目前用的
  }

  // 獲取醫生資訊
  static Future<ManagerModel> getManagerInfo() async {
    final url = Uri.parse('$_baseUrl/manager/Info');
    final token = await AuthService.getToken();

    try {
      final response = await http.post(
        url,
        body: jsonEncode({"token": token}),
        headers: {"Content-Type": "application/json"},
      );
      if (response.statusCode == 200) {
        // ✅ 確保 UTF-8 解析 JSON，防止亂碼
        final utf8DecodedBody = utf8.decode(response.bodyBytes);
        final Map<String, dynamic> data = jsonDecode(utf8DecodedBody);
        return ManagerModel.fromJson(data);
      } else {
        print("Error: ${response.statusCode} - ${response.body}");
      }
    } catch (e) {
      print("Exception: $e");
    }
    return const ManagerModel(); // 發生錯誤時回傳 null
  }

  // 獲取成員資訊
  static Future<MemberModel> getMemberInfo(String memberId) async {
    final url = Uri.parse('$_baseUrl/member/Info?member_id=$memberId');
    final token = await AuthService.getToken();

    try {
      final response = await http.post(
        url,
        body: jsonEncode({"token": token}),
        headers: {"Content-Type": "application/json"},
      );
      if (response.statusCode == 200) {
        // ✅ 確保 UTF-8 解析 JSON，防止亂碼
        final utf8DecodedBody = utf8.decode(response.bodyBytes);
        final Map<String, dynamic> data = jsonDecode(utf8DecodedBody);
        return MemberModel.fromJson(data);
      } else {
        print("Error: ${response.statusCode} - ${response.body}");
      }
    } catch (e) {
      print("Exception: $e");
    }
    return const MemberModel(); // 發生錯誤時回傳 null
  }

  //新增成員
  static Future<bool> memberSignup(MemberModel memberModel, File image) async {
    final url = Uri.parse('$_baseUrl/manager/Member_Signup');
    final token = await AuthService.getToken();

    // 創建 MultipartRequest
    var request = http.MultipartRequest('POST', url);

    // 添加表單欄位（模仿 JavaScript 的 FormData）
    request.fields['id'] = memberModel.id;
    request.fields['sex'] = memberModel.sex;
    request.fields['name'] = memberModel.name;
    request.fields['birthdate'] =
        '${memberModel.yyyy}${memberModel.mm}${memberModel.dd}'; // 格式化為 YYYYMMDD
    request.fields['managerToken'] = token ?? ''; // 如果 token 為 null，給空字串

    // 如果有圖片，添加檔案
    request.files.add(
      await http.MultipartFile.fromPath(
        'profile_image_file', // 與 JavaScript 中的鍵名一致
        image.path,
      ),
    );

    try {
      // 發送請求
      final response = await request.send();

      // 處理回應
      final responseBody = await response.stream.bytesToString(); // 將回應轉為字串
      if (response.statusCode == 200) {
        print("Member signup successful: $responseBody");
        return true;
      } else {
        print("Signup failed: ${response.statusCode} - $responseBody");
        return false;
      }
    } catch (e) {
      print("Exception during signup: $e");
      return false;
    }
  }

  // 新增方法：獲取會員列表
  static Future<List<MemberModel>> getMemberList() async {
    final url = Uri.parse('$_baseUrl/manager/MemberList');
    final token = await AuthService.getToken();

    try {
      final response = await http.post(
        url,
        body: jsonEncode({"token": token}),
        headers: {"Content-Type": "application/json"},
      );

      if (response.statusCode == 200) {
        final utf8DecodedBody = utf8.decode(response.bodyBytes);
        final List<dynamic> data = jsonDecode(utf8DecodedBody);
        return data.map((json) => MemberModel.fromJson(json)).toList();
      } else {
        final utf8DecodedError = utf8.decode(response.bodyBytes);
        print(
          "Fetch member list failed: ${response.statusCode} - $utf8DecodedError",
        );
        return [];
      }
    } catch (e) {
      print("Exception during fetch member list: $e");
      return [];
    }
  }

  // 獲取管理者圖片 URL
  static Future<String?> getManagerImage(String managerId) async {
    final url = Uri.parse('$_baseUrl/manager/Profile/$managerId');
    try {
      final response = await http.get(url);
      if (response.statusCode == 200) {
        // ✅ 如果 API 回傳的是圖片的 URL（例如從後端 S3 或 Cloud Storage）
        return url.toString(); // 回傳圖片 URL
      } else {
        print("Manager image fetch failed: ${response.statusCode}");
      }
    } catch (e) {
      print("Exception fetching manager image: $e");
    }
    return null;
  }

  // 獲取會員圖片 URL
  static Future<String?> getMemberImage(String memberId) async {
    final url = Uri.parse('$_baseUrl/member/Profile/$memberId');
    try {
      final response = await http.get(url);
      if (response.statusCode == 200) {
        return url.toString();
      } else {
        print("Member image fetch failed: ${response.statusCode}");
      }
    } catch (e) {
      print("Exception fetching member image: $e");
    }
    return null;
  }

  // 上傳拍攝紀錄（含MRI檔案與表單欄位）
  static Future<bool> uploadRecord({
    required String memberId,
    required String date, // 格式建議 YYYY-MM-DD
    required File niiFile, // 本地 .nii.gz 檔案
    int? mmseScore, // 可選
  }) async {
    final url = Uri.parse('$_baseUrl/ai/upload/Record');
    final token = await AuthService.getToken();

    if (token == null || token.isEmpty) {
      print("❌ Manager Token 不存在，請重新登入");
      return false;
    }

    final request = http.MultipartRequest('POST', url);

    // === 表單欄位 ===
    request.fields['managerToken'] = token;
    request.fields['member_id'] = memberId;
    request.fields['date'] = date;
    if (mmseScore != null) {
      request.fields['MMSE_score'] = mmseScore.toString();
    }

    // === MRI 檔案上傳 (.nii.gz) ===
    final fileName = niiFile.path.split('/').last;
    request.files.add(
      await http.MultipartFile.fromPath(
        'image_file',
        niiFile.path,
        filename: fileName,
        contentType: MediaType('application', 'gzip'),
      ),
    );

    try {
      final response = await request.send();
      final responseBody = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        print("✅ 上傳紀錄成功：$responseBody");
        return true;
      } else {
        print("❌ 上傳失敗：${response.statusCode} - $responseBody");
        return false;
      }
    } catch (e) {
      print("❌ 上傳時發生例外：$e");
      return false;
    }
  }

  // AI計算
  static Future<Map<String, dynamic>?> runAiPrediction({
    required String memberId,
    required int recordCount,
  }) async {
    final token = await AuthService.getToken();

    if (token == null || token.isEmpty) {
      print("❌ Manager Token 不存在，請重新登入");
      return null;
    }

    final url = Uri.parse(
      '$_baseUrl/ai/$memberId',
    ).replace(queryParameters: {'record_count': recordCount.toString()});

    try {
      final response = await http.post(
        url,
        body: {'manager_token': token},
        headers: {"Content-Type": "application/x-www-form-urlencoded"},
      );

      if (response.statusCode == 200) {
        final utf8Body = utf8.decode(response.bodyBytes);
        final Map<String, dynamic> data = jsonDecode(utf8Body);
        print("✅ AI 預測成功：$data");
        return data;
      } else {
        final utf8Body = utf8.decode(response.bodyBytes);
        print("❌ AI 預測失敗：${response.statusCode} - $utf8Body");
        return null;
      }
    } catch (e) {
      print("❌ 發生例外：$e");
      return null;
    }
  }

  // 呼叫 2D 切片並儲存結果
  static Future<bool> sliceAndStoreMRI({
    required String memberId,
    required int recordCount,
  }) async {
    final token = await AuthService.getToken();

    if (token == null || token.isEmpty) {
      print("❌ Manager Token 不存在，請重新登入");
      return false;
    }

    final url = Uri.parse(
      '$_baseUrl/ai/restore/$memberId?record_count=$recordCount',
    );

    try {
      final response = await http.post(
        url,
        body: {'record_count': recordCount.toString(), 'manager_token': token},
      );

      if (response.statusCode == 200) {
        print("✅ MRI 切片與儲存完成");
        return true;
      } else {
        final utf8Body = utf8.decode(response.bodyBytes);
        print("❌ 切片失敗：${response.statusCode} - $utf8Body");
        return false;
      }
    } catch (e) {
      print("❌ 切片時發生例外：$e");
      return false;
    }
  }

  // 取得紀錄表
  static Future<List<RecordModel>> getMemberRecordList(String memberId) async {
    final token = await AuthService.getToken();

    if (token == null || token.isEmpty) {
      print("❌ Token 不存在，請重新登入");
      return [];
    }

    final url = Uri.parse('$_baseUrl/member/RecordsList?member_id=$memberId');

    try {
      final response = await http.post(
        url,
        body: jsonEncode({"token": token}),
        headers: {"Content-Type": "application/json"},
      );

      if (response.statusCode == 200) {
        final utf8Body = utf8.decode(response.bodyBytes);
        final List<dynamic> jsonList = jsonDecode(utf8Body);

        return jsonList
            .map((json) => RecordModel.fromJson(json as Map<String, dynamic>))
            .toList();
      } else {
        final utf8Body = utf8.decode(response.bodyBytes);
        print("❌ 成員紀錄列表取得失敗：${response.statusCode} - $utf8Body");
        return [];
      }
    } catch (e) {
      print("❌ 發生例外：$e");
      return [];
    }
  }

  // 取得切片
  static Future<Map<String, List<File>>> fetchAndUnzipSlices(
    String memberId,
    int recordCount,
  ) async {
    final url = Uri.parse('$_baseUrl/ai/slice/all/$memberId/$recordCount');
    final response = await http.get(url);

    if (response.statusCode != 200) {
      throw Exception('下載失敗: ${response.statusCode}');
    }

    // 解析 JSON 結構
    final Map<String, dynamic> jsonData = jsonDecode(
      utf8.decode(response.bodyBytes),
    );

    final axial = <File>[];
    final coronal = <File>[];
    final sagittal = <File>[];
    final gradCAM_axial = <File>[];
    final gradCAM_coronal = <File>[];
    final gradCAM_sagittal = <File>[];

    final tempDir = await getTemporaryDirectory();
    final folderPath = '${tempDir.path}\\$memberId-$recordCount';

    // ✅ 建立資料夾（若不存在）
    final dir = Directory(folderPath);
    if (!(await dir.exists())) {
      await dir.create(recursive: true);
    }
    // 工具函數：處理每一切面
    Future<void> saveImages(
      String plane,
      List<dynamic> base64List,
      List<File> outputList,
    ) async {
      for (int i = 0; i < base64List.length; i++) {
        final b64 = base64List[i];
        final bytes = base64Decode(b64);
        final path = '$folderPath/${plane}_$i.png';
        final file = File(path);
        await file.writeAsBytes(bytes);
        outputList.add(file);
      }
    }

    // 根據每一個切面解碼 & 儲存
    await saveImages('axial', jsonData['axial'] ?? [], axial);
    await saveImages('coronal', jsonData['coronal'] ?? [], coronal);
    await saveImages('sagittal', jsonData['sagittal'] ?? [], sagittal);
    await saveImages(
      'gradCAM_axial',
      jsonData['gradCAM_axial'] ?? [],
      gradCAM_axial,
    );
    await saveImages(
      'gradCAM_coronal',
      jsonData['gradCAM_coronal'] ?? [],
      gradCAM_coronal,
    );
    await saveImages(
      'gradCAM_sagittal',
      jsonData['gradCAM_sagittal'] ?? [],
      gradCAM_sagittal,
    );
    return {
      'axial': axial,
      'coronal': coronal,
      'sagittal': sagittal,
      'gradCAM_axial': gradCAM_axial,
      'gradCAM_coronal': gradCAM_coronal,
      'gradCAM_sagittal': gradCAM_sagittal,
    };
  }
}
