#include <flutter/dart_project.h>
#include <flutter/flutter_view_controller.h>
#include <windows.h>

#include "flutter_window.h"
#include "utils.h"

#include <dwmapi.h>                // DWM (Desktop Window Manager) API
#pragma comment(lib, "dwmapi.lib") // 記得連結 dwnapi.lib，不然編譯會錯

int APIENTRY wWinMain(_In_ HINSTANCE instance, _In_opt_ HINSTANCE prev,
                      _In_ wchar_t *command_line, _In_ int show_command)
{
  // Attach to console when present (e.g., 'flutter run') or create a
  // new console when running with a debugger.
  if (!::AttachConsole(ATTACH_PARENT_PROCESS) && ::IsDebuggerPresent())
  {
    CreateAndAttachConsole();
  }

  // Initialize COM, so that it is available for use in the library and/or
  // plugins.
  ::CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

  flutter::DartProject project(L"data");

  std::vector<std::string> command_line_arguments =
      GetCommandLineArguments();

  project.set_dart_entrypoint_arguments(std::move(command_line_arguments));

  FlutterWindow window(project);
  Win32Window::Point origin(10, 10);
  Win32Window::Size size(1280, 720);
  if (!window.Create(L"Trajectory", origin, size))
  {
    return EXIT_FAILURE;
  }
  // === 加在這裡開始設定標題列顏色 ===

  // 想要的顏色，這邊以綠色 (RGB: 0, 128, 0) 為例
  COLORREF titleColor = RGB(15, 27, 30);

  // 啟用自訂標題列顏色
  BOOL useImmersiveDarkMode = TRUE;
  DwmSetWindowAttribute(window.GetHandle(), DWMWA_USE_IMMERSIVE_DARK_MODE, &useImmersiveDarkMode, sizeof(useImmersiveDarkMode));

  // 設定標題列背景色
  DwmSetWindowAttribute(window.GetHandle(), DWMWA_CAPTION_COLOR, &titleColor, sizeof(titleColor));

  // 如果想改文字顏色（例如白色文字）
  COLORREF textColor = RGB(255, 255, 255);
  DwmSetWindowAttribute(window.GetHandle(), DWMWA_TEXT_COLOR, &textColor, sizeof(textColor));
  window.SetQuitOnClose(true);

  ::MSG msg;
  while (::GetMessage(&msg, nullptr, 0, 0))
  {
    ::TranslateMessage(&msg);
    ::DispatchMessage(&msg);
  }

  ::CoUninitialize();
  return EXIT_SUCCESS;
}
