/**
 * Download helper — triggers browser file download via Blob + object URL.
 */

/**
 * Download content as a file.
 *
 * @param content - File content as string
 * @param filename - Desired filename (including extension)
 * @param mimeType - MIME type of the content
 */
export function downloadFile(content: string, filename: string, mimeType: string): void {
  try {
    const blob = new Blob([content], { type: mimeType })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.style.display = 'none'
    document.body.appendChild(a)
    a.click()

    // Cleanup
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  } catch (error) {
    console.error('[downloadHelper] downloadFile failed:', error)
  }
}

/**
 * Open printable HTML in a new window and trigger print dialog.
 *
 * @param htmlContent - Full HTML document string
 */
export function printHtml(htmlContent: string): void {
  try {
    const printWindow = window.open('', '_blank', 'width=800,height=600')
    if (!printWindow) {
      console.error('[downloadHelper] printHtml: Popup blocked')
      return
    }
    printWindow.document.write(htmlContent)
    printWindow.document.close()

    // Wait for resources to load, then print
    printWindow.onload = () => {
      printWindow.focus()
      printWindow.print()
    }
  } catch (error) {
    console.error('[downloadHelper] printHtml failed:', error)
  }
}
