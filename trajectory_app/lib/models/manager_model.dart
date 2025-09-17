class ManagerModel {
  final String id;
  final String department;
  final String name;
  final String numMembers;

  const ManagerModel({
    this.id = '--',
    this.name = '--',
    this.department = '--',
    this.numMembers = '--',
  });
  // **從 JSON 轉換為 Model**
  factory ManagerModel.fromJson(Map<String, dynamic> json) {
    return ManagerModel(
      id: json['id'] ?? '--',
      department: json['department'] ?? '--',
      name: json['name'] ?? '--',
      numMembers: json['numMembers']?.toString() ?? '--', // 確保轉為字串
    );
  }
}
