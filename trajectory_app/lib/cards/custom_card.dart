import 'package:trajectory_app/const/constant.dart';
import 'package:flutter/material.dart';

class CustomCard extends StatelessWidget {
  final Widget child;
  final Color? color;
  final Color? borderColor;
  final EdgeInsetsGeometry? padding;
  final double? width;
  final double? height;

  const CustomCard({
    super.key,
    this.color,
    this.borderColor,
    this.padding,
    this.width,
    this.height,
    required this.child,
  });
  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: width,
      height: height,
      child: Container(
        decoration: BoxDecoration(
          borderRadius: const BorderRadius.all(Radius.circular(8.0)),
          color: color ?? cardBackgroundColor,
          border: Border.all(
            color: borderColor ?? Colors.transparent,
            width: 2,
          ),
        ),
        child: Padding(
          padding: padding ?? const EdgeInsets.all(12.0),
          child: child,
        ),
      ),
    );
  }
}
