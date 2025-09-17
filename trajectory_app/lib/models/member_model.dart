class MemberModel {
  final String id;
  final String sex;
  final String name;
  final String yyyy;
  final String mm;
  final String dd;
  final String numRecords;

  const MemberModel({
    this.id = '--',
    this.sex = '--',
    this.name = '--',
    this.yyyy = '--',
    this.mm = '--',
    this.dd = '--',
    this.numRecords = '--',
  });
  // 從 JSON 創建 MemberModel
  factory MemberModel.fromJson(Map<String, dynamic> json) {
    String yyyy = '--';
    String mm = '--';
    String dd = '--';
    if (json['birthdate'] != null && json['birthdate'].length == 8) {
      final birthdayStr = json['birthdate'];
      yyyy = birthdayStr.substring(0, 4);
      mm = birthdayStr.substring(4, 6);
      dd = birthdayStr.substring(6, 8);
    }
    return MemberModel(
      id: json['id'] as String? ?? '--',
      sex: json['sex'] as String? ?? '--',
      name: json['name'] as String? ?? '--',
      yyyy: yyyy,
      mm: mm,
      dd: dd,
      numRecords:
          json['record_count'] != null ? json['record_count'].toString() : '--',
    );
  }
}
