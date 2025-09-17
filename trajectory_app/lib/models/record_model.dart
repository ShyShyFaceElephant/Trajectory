class RecordModel {
  final String yyyy;
  final String mm;
  final String dd;
  final String brainAge;
  final String actualAge;
  final String mmseScore;
  final String riskScore;

  const RecordModel({
    required this.yyyy,
    required this.mm,
    required this.dd,
    required this.brainAge,
    required this.actualAge,
    required this.mmseScore,
    required this.riskScore,
  });

  factory RecordModel.fromJson(Map<String, dynamic> json) {
    String yyyy = '--';
    String mm = '--';
    String dd = '--';

    if (json['date'] != null && json['date'].length == 8) {
      final dateStr = json['date'];
      yyyy = dateStr.substring(0, 4);
      mm = dateStr.substring(4, 6);
      dd = dateStr.substring(6, 8);
    }

    return RecordModel(
      yyyy: yyyy,
      mm: mm,
      dd: dd,
      brainAge: json['brain_age']?.toString() ?? '--',
      actualAge: json['actual_age']?.toString() ?? '--',
      mmseScore: json['MMSE_score']?.toString() ?? '--',
      riskScore:
          (json['risk_score']?.toString().isEmpty ?? true)
              ? '--'
              : json['risk_score'].toString(),
    );
  }
}
