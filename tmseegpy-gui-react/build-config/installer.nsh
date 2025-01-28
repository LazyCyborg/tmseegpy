!macro preInit
    SetRegView 64
    WriteRegExpandStr HKLM "${INSTALL_REGISTRY_KEY}" InstallLocation "$PROGRAMFILES64\${PRODUCT_NAME}"
    WriteRegExpandStr HKCU "${INSTALL_REGISTRY_KEY}" InstallLocation "$PROGRAMFILES64\${PRODUCT_NAME}"
    SetRegView 32
    WriteRegExpandStr HKLM "${INSTALL_REGISTRY_KEY}" InstallLocation "$PROGRAMFILES\${PRODUCT_NAME}"
    WriteRegExpandStr HKCU "${INSTALL_REGISTRY_KEY}" InstallLocation "$PROGRAMFILES\${PRODUCT_NAME}"
!macroend

!macro customInit
    # Add any custom initialization here
!macroend

!macro customInstall
    # Environment variables
    ${EnvVarUpdate} $0 "PATH" "A" "HKLM" "$INSTDIR\resources\python"

    # Create application shortcut in Start Menu
    CreateDirectory "$SMPROGRAMS\${PRODUCT_NAME}"
    CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME}.lnk" "$INSTDIR\${PRODUCT_FILENAME}"
    CreateShortCut "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall ${PRODUCT_NAME}.lnk" "$INSTDIR\Uninstall ${PRODUCT_NAME}.exe"

    # Create desktop shortcut
    CreateShortCut "$DESKTOP\${PRODUCT_NAME}.lnk" "$INSTDIR\${PRODUCT_FILENAME}"

    # Add application to Windows' Programs and Features
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" \
                     "DisplayName" "${PRODUCT_NAME}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" \
                     "UninstallString" "$INSTDIR\Uninstall ${PRODUCT_NAME}.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" \
                     "DisplayIcon" "$INSTDIR\${PRODUCT_FILENAME}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" \
                     "Publisher" "${PUBLISHER}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" \
                     "DisplayVersion" "${VERSION}"
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" \
                       "NoModify" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" \
                       "NoRepair" 1
!macroend

!macro customUnInstall
    # Remove Start Menu shortcuts
    RMDir /r "$SMPROGRAMS\${PRODUCT_NAME}"

    # Remove desktop shortcut
    Delete "$DESKTOP\${PRODUCT_NAME}.lnk"

    # Remove environment variables
    ${un.EnvVarUpdate} $0 "PATH" "R" "HKLM" "$INSTDIR\resources\python"

    # Remove registry entries
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"

    # Clean up program files
    RMDir /r "$INSTDIR"

    # Remove any remaining user data (optional - uncomment if needed)
    # RMDir /r "$APPDATA\${PRODUCT_NAME}"
!macroend

# Optional: Add file association macros if needed
!macro registerFileAssociations
    WriteRegStr HKCR ".ses" "" "${PRODUCT_NAME}.ses"
    WriteRegStr HKCR "${PRODUCT_NAME}.ses" "" "TMSeegpy Session File"
    WriteRegStr HKCR "${PRODUCT_NAME}.ses\DefaultIcon" "" "$INSTDIR\${PRODUCT_FILENAME},0"
    WriteRegStr HKCR "${PRODUCT_NAME}.ses\shell\open\command" "" '"$INSTDIR\${PRODUCT_FILENAME}" "%1"'
!macroend

!macro unregisterFileAssociations
    DeleteRegKey HKCR ".ses"
    DeleteRegKey HKCR "${PRODUCT_NAME}.ses"
!macroend

# Optional: Add custom error handling
!macro exitFailure
    MessageBox MB_OK|MB_ICONSTOP "Installation failed. Please check the logs for more information."
    Quit
!macroend