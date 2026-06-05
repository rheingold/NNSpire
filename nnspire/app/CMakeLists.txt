# NNSpire Qt application — GPL v3
# Phase 3

if(NN_QT_VERSION EQUAL 6)
    qt_add_executable(NNSpire
        # main.cpp
    )
    qt_add_qml_module(NNSpire
        URI "NNSpire"
        VERSION 1.0
        QML_FILES
            # ui/MainWindow.qml
            # ui/ModelEditor.qml
            # ui/TrainingDashboard.qml
            # ui/WeightViewer.qml
            # ui/NeuronViewer.qml
            # ui/TokenizerPanel.qml
            # ui/PipelinePanel.qml
            # ui/DeploymentPanel.qml
            # ui/QuantumPanel.qml
            # ui/HelpPanel.qml
            # ui/WizardMode.qml
        SOURCES
            # controllers/ModelController.h
            # controllers/ModelController.cpp
            # controllers/TrainingController.h
            # controllers/TrainingController.cpp
            # controllers/BackendController.h
            # controllers/BackendController.cpp
            # controllers/PluginController.h
            # controllers/PluginController.cpp
            # controllers/HelpController.h
            # controllers/HelpController.cpp
    )
else()
    add_executable(NNSpire
        # main.cpp
    )
endif()

set_target_properties(NNSpire PROPERTIES
    WIN32_EXECUTABLE TRUE
    MACOSX_BUNDLE    TRUE
)

target_link_libraries(NNSpire PRIVATE
    NNSpire-core
    NNSpire-trust
    NNSpire-pipeline
    NNSpire-deployment
    Qt::Core Qt::Gui Qt::Widgets Qt::Qml Qt::Quick Qt::Charts Qt::Network Qt::Concurrent
)

# Qt resource: AI Standards KB (for in-app help)
# qt_add_resources(NNSpire "kb"
#     PREFIX "/kb"
#     BASE "${CMAKE_SOURCE_DIR}/../ai-standards-kb"
#     FILES ...
# )

# ─── Platform deployment helpers ───────────────────────────────────────────────
if(WIN32)
    add_custom_command(TARGET NNSpire POST_BUILD
        COMMAND "${Qt_DIR}/../../../bin/windeployqt.exe"
                --qmldir "${CMAKE_CURRENT_SOURCE_DIR}/ui"
                "$<TARGET_FILE:NNSpire>"
        COMMENT "Running windeployqt"
    )
elseif(APPLE)
    set_target_properties(NNSpire PROPERTIES
        MACOSX_BUNDLE_GUI_IDENTIFIER "dev.NNSpire.NNSpire"
        MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
    )
endif()
