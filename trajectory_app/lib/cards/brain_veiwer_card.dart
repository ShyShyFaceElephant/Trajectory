import 'dart:async';
import 'dart:io';

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:trajectory_app/cards/custom_card.dart';
import 'package:trajectory_app/models/record_model.dart';
import 'package:trajectory_app/services/api_service.dart';

class BrainViewerCard extends StatefulWidget {
  final String memberId;
  final RecordModel record;
  final int recordIndex;
  final void Function(int) popBrainViewer;

  const BrainViewerCard({
    super.key,
    required this.record,
    required this.recordIndex,
    required this.popBrainViewer,
    required this.memberId,
  });

  @override
  State<BrainViewerCard> createState() => _BrainViewerCardState();
}

class _BrainViewerCardState extends State<BrainViewerCard> {
  Map<String, List<File>>? imageSet;
  final Map<String, int> sliceIndex = {'axial': 0, 'coronal': 0, 'sagittal': 0};
  final Map<String, int> sliceMax = {'axial': 0, 'coronal': 0, 'sagittal': 0};
  bool showGradCAM = false;
  Timer? _longPressTimer;

  @override
  void initState() {
    super.initState();
    _loadSlices();
  }

  Future<void> _loadSlices() async {
    try {
      final result = await ApiService.fetchAndUnzipSlices(
        widget.memberId,
        widget.recordIndex + 1,
      );

      for (final plane in [
        'axial',
        'coronal',
        'sagittal',
        'gradCAM_axial',
        'gradCAM_coronal',
        'gradCAM_sagittal',
      ]) {
        final imageList = result[plane] ?? [];
        for (final file in imageList) {
          await precacheImage(FileImage(file), context);
        }
      }

      if (!mounted) return;
      setState(() {
        imageSet = result;
        for (var plane in ['axial', 'coronal', 'sagittal']) {
          sliceMax[plane] = result[plane]?.length ?? 0;
          sliceIndex[plane] = (sliceMax[plane]! / 2).toInt();
        }
      });
    } catch (e) {
      print("\u274c \u8f09\u5165\u5207\u7247\u5931\u6557: $e");
    }
  }

  void _updateSlice(String label, bool isIncrement) {
    setState(() {
      final value = sliceIndex[label]!;
      final max = sliceMax[label]!;
      if (isIncrement && value < max - 1) {
        sliceIndex[label] = value + 1;
      } else if (!isIncrement && value > 0) {
        sliceIndex[label] = value - 1;
      }
    });
  }

  void _startAutoStep(String plane, bool isIncrement) {
    _longPressTimer = Timer.periodic(const Duration(milliseconds: 20), (_) {
      _updateSlice(plane, isIncrement);
    });
  }

  void _stopAutoStep() {
    _longPressTimer?.cancel();
    _longPressTimer = null;
  }

  @override
  Widget build(BuildContext context) {
    return CustomCard(
      padding: const EdgeInsets.all(20),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const SizedBox(width: 40),
              Column(
                children: [
                  Text(
                    "${widget.record.yyyy}年${widget.record.mm}月${widget.record.dd}日",
                    style: const TextStyle(fontSize: 16, color: Colors.white),
                  ),
                  Text(
                    "實際年齡 / 腦部年齡: ${widget.record.actualAge} / ${widget.record.brainAge} 歲",
                    style: const TextStyle(fontSize: 14, color: Colors.white),
                  ),
                ],
              ),
              IconButton(
                onPressed: () => widget.popBrainViewer(widget.recordIndex),
                icon: const Icon(Icons.close, color: Colors.white, size: 24),
              ),
            ],
          ),
          Row(
            children: [
              Checkbox(
                value: showGradCAM,
                onChanged: (value) {
                  setState(() {
                    showGradCAM = value ?? false;
                  });
                },
              ),
              const Text("顯示關鍵腦區", style: TextStyle(color: Colors.white70)),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children:
                ['axial', 'coronal', 'sagittal'].map((plane) {
                  return SliceViewer(
                    label: plane,
                    normalImages: imageSet?[plane],
                    gradCAMImages: imageSet?['gradCAM_$plane'],
                    index: sliceIndex[plane]!,
                    showGradCAM: showGradCAM,
                    onStep: (isIncrement) => _updateSlice(plane, isIncrement),
                    onLongPressStartIncrement:
                        () => _startAutoStep(plane, true),
                    onLongPressStartDecrement:
                        () => _startAutoStep(plane, false),
                    onLongPressEnd: _stopAutoStep,
                  );
                }).toList(),
          ),
        ],
      ),
    );
  }
}

class SliceViewer extends StatelessWidget {
  final String label;
  final List<File>? normalImages;
  final List<File>? gradCAMImages;
  final int index;
  final bool showGradCAM;
  final void Function(bool isIncrement) onStep;
  final void Function() onLongPressStartIncrement;
  final void Function() onLongPressStartDecrement;
  final void Function() onLongPressEnd;

  const SliceViewer({
    super.key,
    required this.label,
    required this.normalImages,
    required this.gradCAMImages,
    required this.index,
    required this.showGradCAM,
    required this.onStep,
    required this.onLongPressStartIncrement,
    required this.onLongPressStartDecrement,
    required this.onLongPressEnd,
  });

  @override
  Widget build(BuildContext context) {
    final currentImages = showGradCAM ? gradCAMImages : normalImages;
    final currentImage =
        (currentImages != null && index < currentImages.length)
            ? currentImages[index]
            : null;

    return Column(
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
        const SizedBox(height: 8),
        Listener(
          onPointerSignal: (pointerSignal) {
            if (pointerSignal is PointerScrollEvent) {
              if (pointerSignal.scrollDelta.dy < 0) {
                onStep(false);
              } else if (pointerSignal.scrollDelta.dy > 0) {
                onStep(true);
              }
            }
          },
          child: Container(
            width: 230,
            height: 230,
            color: Colors.black,
            child:
                currentImage != null
                    ? Image.file(
                      currentImage,
                      fit: BoxFit.cover,
                      gaplessPlayback: true,
                    )
                    : const Center(
                      child: CircularProgressIndicator(color: Colors.white),
                    ),
          ),
        ),
        const SizedBox(height: 8),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            GestureDetector(
              onLongPressStart: (_) => onLongPressStartDecrement(),
              onLongPressEnd: (_) => onLongPressEnd(),
              child: IconButton(
                onPressed: () => onStep(false),
                icon: const Icon(Icons.remove, color: Colors.white),
                style: IconButton.styleFrom(backgroundColor: Colors.grey[800]),
              ),
            ),
            const SizedBox(width: 8),
            SizedBox(
              width: 60,
              child: Center(
                child: Text(
                  index.toString(),
                  style: const TextStyle(fontSize: 18, color: Colors.white),
                ),
              ),
            ),
            const SizedBox(width: 8),
            GestureDetector(
              onLongPressStart: (_) => onLongPressStartIncrement(),
              onLongPressEnd: (_) => onLongPressEnd(),
              child: IconButton(
                onPressed: () => onStep(true),
                icon: const Icon(Icons.add, color: Colors.white),
                style: IconButton.styleFrom(backgroundColor: Colors.grey[800]),
              ),
            ),
          ],
        ),
      ],
    );
  }
}
