import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@workspace/ui/components/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@workspace/ui/components/tabs"
import { Input } from "@workspace/ui/components/input"
import { Label } from "@workspace/ui/components/label"
import { Button } from "@workspace/ui/components/button"
import { Loader2, Upload, BarChart3, Image as ImageIcon, FileText } from "lucide-react"

import { RealTimePathFinder } from "./components/RealTimePathFinder"

const BACKEND_URL = "http://localhost:8000"

const CLASS_COLORS: Record<string, string> = {
  "Trees": "#228B22",
  "Lush Bushes": "#00FF00",
  "Dry Grass": "#D2B48C",
  "Dry Bushes": "#8B5A2B",
  "Ground Clutter": "#808000",
  "Flowers": "#FF69B4",
  "Logs": "#8B4513",
  "Rocks": "#808080",
  "Landscape": "#A0522D",
  "Sky": "#87CEEB",
}

export function App() {
  const [loading, setLoading] = useState(false)
  
  // Single Inference State
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [resultImages, setResultImages] = useState<{ original: string, mask: string, path: string } | null>(null)
  
  // Detailed Report State
  const [reportFiles, setReportFiles] = useState<{ image: File | null, gt: File | null }>({ image: null, gt: null })
  const [reportData, setReportData] = useState<any | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)


  const handleSingleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setPreviewUrl(URL.createObjectURL(file))
      setResultImages(null)
      handleUpload(file)
    }
  }

  const handleUpload = async (file: File) => {
    setLoading(true)
    setErrorMsg(null)
    const formData = new FormData()
    formData.append("file", file)

    try {
      const response = await fetch(`${BACKEND_URL}/predict`, {
        method: "POST",
        body: formData,
      })
      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || "Inference failed")
      }
      const data = await response.json()
      setResultImages({
        original: `data:image/jpeg;base64,${data.original}`,
        mask: `data:image/jpeg;base64,${data.mask}`,
        path: `data:image/jpeg;base64,${data.path_viz}`
      })
    } catch (err: any) {
      console.error("Inference error:", err)
      setErrorMsg(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleAnalyze = async () => {
    if (!reportFiles.image || !reportFiles.gt) return
    
    setLoading(true)
    setErrorMsg(null)
    const formData = new FormData()
    formData.append("colour_image", reportFiles.image)
    formData.append("segment_image", reportFiles.gt)

    try {
      const response = await fetch(`${BACKEND_URL}/analyze`, {
        method: "POST",
        body: formData,
      })
      if (!response.ok) {
        const error = await response.json()
        const detail = typeof error.detail === 'string' 
          ? error.detail 
          : JSON.stringify(error.detail, null, 2)
        throw new Error(detail || "Report generation failed")
      }
      const data = await response.json()
      setReportData(data)
    } catch (err: any) {
      console.error("Report generation error:", err)
      setErrorMsg(err.message)
    } finally {
      setLoading(false)
    }
  }

  const getPlotUrl = (name: string) => `${BACKEND_URL}/static/plots/${name}`

  return (
    <div className="min-h-screen bg-[#0a0a0c] p-8 text-slate-100 font-sans selection:bg-indigo-500/30">
      <div className="mx-auto max-w-7xl space-y-8">
        <header className="space-y-2 flex justify-between items-end">
          <div>
            <h1 className="text-5xl font-extrabold tracking-tighter text-white mt-4">
              Offroad <span className="text-indigo-400 italic">Segmentation</span> Explorer
            </h1>
            <p className="text-xl text-slate-400 font-light">Precision analytics for rugged terrain perception.</p>
          </div>
          <div className="text-right hidden md:block">
            <p className="text-xs font-bold uppercase tracking-widest text-slate-600">System Status</p>
            <p className="text-sm font-medium flex items-center gap-2 justify-end text-slate-300">
              <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
              Backend Online
            </p>
          </div>
        </header>

        {errorMsg && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded-lg shadow-sm animate-in fade-in slide-in-from-top-4">
             <div className="flex items-center gap-3">
               <div className="h-2 w-2 rounded-full bg-red-500"></div>
               <p className="text-sm font-bold text-red-800 uppercase tracking-widest">Error detected</p>
             </div>
             <p className="text-red-700 mt-1 ml-5 font-medium">{errorMsg}</p>
          </div>
        )}

        <Tabs defaultValue="report" className="w-full">
          <TabsList className="grid w-full max-w-2xl grid-cols-6 p-1 bg-slate-900/60 backdrop-blur-sm rounded-xl border border-white/5">
            <TabsTrigger value="report" className="rounded-lg data-[state=active]:bg-slate-800 data-[state=active]:text-white text-slate-400 data-[state=active]:shadow-lg transition-all">Full Report</TabsTrigger>
            <TabsTrigger value="realtime" className="rounded-lg data-[state=active]:bg-slate-800 data-[state=active]:text-white text-slate-400 data-[state=active]:shadow-lg transition-all">Real-Time</TabsTrigger>
            <TabsTrigger value="inference" className="rounded-lg data-[state=active]:bg-slate-800 data-[state=active]:text-white text-slate-400 data-[state=active]:shadow-lg transition-all">Quick Inference</TabsTrigger>
            <TabsTrigger value="metrics" className="rounded-lg data-[state=active]:bg-slate-800 data-[state=active]:text-white text-slate-400 data-[state=active]:shadow-lg transition-all">Metrics</TabsTrigger>
            <TabsTrigger value="iou" className="rounded-lg data-[state=active]:bg-slate-800 data-[state=active]:text-white text-slate-400 data-[state=active]:shadow-lg transition-all">IoU</TabsTrigger>
            <TabsTrigger value="matrix" className="rounded-lg data-[state=active]:bg-slate-800 data-[state=active]:text-white text-slate-400 data-[state=active]:shadow-lg transition-all">Confusion</TabsTrigger>
            <TabsTrigger value="dist" className="rounded-lg data-[state=active]:bg-slate-800 data-[state=active]:text-white text-slate-400 data-[state=active]:shadow-lg transition-all">Dataset</TabsTrigger>
          </TabsList>

          <TabsContent value="report" className="mt-8">
            <Card className="border border-white/5 shadow-2xl bg-slate-900/40 backdrop-blur-xl overflow-hidden ring-1 ring-white/5">
              <CardHeader className="border-b border-white/5 bg-white/[0.02]">
                <div className="flex justify-between items-center">
                  <div>
                    <CardTitle className="text-2xl font-bold text-white">Segmentations Inference Report</CardTitle>
                    <CardDescription className="text-slate-400">Generate detailed performance metrics with comparison.</CardDescription>
                  </div>
                  <Button 
                    onClick={handleAnalyze} 
                    disabled={!reportFiles.image || !reportFiles.gt || loading}
                    className="gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-8 py-7 h-auto text-lg font-bold rounded-2xl transition-all active:scale-95 shadow-[0_10px_25px_-5px_rgba(79,70,229,0.4)] hover:shadow-[0_15px_30px_-5px_rgba(79,70,229,0.6)] disabled:opacity-50 disabled:shadow-none"
                  >
                    {loading ? <Loader2 className="h-6 w-6 animate-spin" /> : <BarChart3 className="h-6 w-6" />}
                    Generate Analytics
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="p-0">
                <div className="grid grid-cols-1 lg:grid-cols-4 divide-x divide-white/5">
                  <div className="p-8 space-y-8 bg-black/20">
                    <div className="space-y-4">
                      <Label htmlFor="img-upload" className="flex items-center gap-2 text-sm font-bold text-slate-400 uppercase tracking-wider">
                        <ImageIcon className="h-4 w-4" /> Colour Image
                      </Label>
                      <div className={`relative group border-2 border-dashed rounded-2xl p-4 transition-all duration-300 ${reportFiles.image ? 'border-blue-500/50 bg-blue-500/5 shadow-[0_0_20px_rgba(59,130,246,0.2)]' : 'border-white/10 hover:border-blue-500/40 hover:bg-white/[0.02]'}`}>
                        <Input 
                          id="img-upload" 
                          type="file" 
                          accept="image/*" 
                          className="absolute inset-0 opacity-0 cursor-pointer z-10" 
                          onChange={(e) => setReportFiles(p => ({ ...p, image: e.target.files?.[0] || null }))}
                        />
                        <div className="text-center py-4">
                          {reportFiles.image ? (
                            <p className="text-sm font-bold truncate text-blue-600 drop-shadow-sm">{reportFiles.image.name}</p>
                          ) : (
                            <div className="space-y-1">
                              <p className="text-sm text-slate-400 group-hover:text-blue-500 transition-colors">Drop or click colour image</p>
                              <p className="text-[10px] text-slate-300 uppercase tracking-tighter">Raw Input</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <Label htmlFor="gt-upload" className="flex items-center gap-2 text-sm font-bold text-slate-400 uppercase tracking-wider">
                        <FileText className="h-4 w-4" /> Segment Image
                      </Label>
                      <div className={`relative group border-2 border-dashed rounded-2xl p-4 transition-all duration-300 ${reportFiles.gt ? 'border-emerald-500/50 bg-emerald-500/5 shadow-[0_0_20px_rgba(16,185,129,0.2)]' : 'border-white/10 hover:border-emerald-500/40 hover:bg-white/[0.02]'}`}>
                        <Input 
                          id="gt-upload" 
                          type="file" 
                          accept="image/*" 
                          className="absolute inset-0 opacity-0 cursor-pointer z-10" 
                          onChange={(e) => setReportFiles(p => ({ ...p, gt: e.target.files?.[0] || null }))}
                        />
                        <div className="text-center py-4">
                          {reportFiles.gt ? (
                            <p className="text-sm font-bold truncate text-emerald-600 drop-shadow-sm">{reportFiles.gt.name}</p>
                          ) : (
                            <div className="space-y-1">
                              <p className="text-sm text-slate-400 group-hover:text-emerald-500 transition-colors">Drop or click segment image</p>
                              <p className="text-[10px] text-slate-300 uppercase tracking-tighter">Ground Truth</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>

                      <div className="pt-8 space-y-6">
                        <h4 className="text-sm font-bold text-white uppercase tracking-widest border-b border-white/5 pb-2">Inference Summary</h4>
                        <div className="space-y-4">
                          <div className="flex justify-between text-sm">
                            <span className="text-slate-500">Model Architecture</span>
                            <span className="font-mono font-bold text-slate-300">DeepLabV3+</span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-slate-500">Encoder</span>
                            <span className="font-mono font-bold text-indigo-400">ResNet50</span>
                          </div>
                          <div className="flex justify-between text-sm">
                            <span className="text-slate-500">Input Size</span>
                            <span className="font-mono font-bold text-slate-300">320x320</span>
                          </div>
                        </div>
                        
                        {reportData && (
                          <div className="space-y-2 pt-4">
                             <p className="text-[10px] font-bold text-slate-600 uppercase tracking-tighter">Class Distribution (% of image):</p>
                             <div className="space-y-1.5 overflow-hidden">
                                {Object.entries(reportData.stats)
                                  .sort(([,a]: any, [,b]: any) => b.percentage - a.percentage)
                                  .map(([cls, stat]: [string, any]) => (
                                    <div key={cls} className="flex items-center gap-2 text-[11px] font-mono leading-none">
                                      <span className="w-24 truncate text-slate-500">{cls.padEnd(14, '.')}</span>
                                      <span className="font-bold text-slate-200">{stat.percentage.toFixed(2).padStart(5)}%</span>
                                    </div>
                                  ))}
                             </div>
                          </div>
                        )}
                      </div>
                  </div>

                  <div className="lg:col-span-3 p-8">
                    {!reportData ? (
                      <div className="h-full flex flex-col items-center justify-center text-slate-700 space-y-4 py-40">
                        <div className="p-8 bg-white/[0.02] rounded-full ring-1 ring-white/5">
                          <ImageIcon className="h-16 w-16 opacity-10" />
                        </div>
                        <p className="text-xl font-medium text-slate-500">Upload Images to generate analysis</p>
                      </div>
                    ) : (
                      <div className="space-y-12">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                          <div className="space-y-3">
                            <p className="text-xs font-bold text-slate-600 uppercase tracking-widest text-center">Original Image</p>
                            <img src={`data:image/jpeg;base64,${reportData.original}`} className="w-full rounded-2xl shadow-2xl ring-1 ring-white/10" alt="Original" />
                          </div>
                          <div className="space-y-3">
                            <p className="text-xs font-bold text-slate-600 uppercase tracking-widest text-center">Predicted Segmentation</p>
                            <img src={`data:image/jpeg;base64,${reportData.prediction}`} className="w-full rounded-2xl shadow-2xl ring-1 ring-white/10" alt="Prediction" />
                          </div>
                          <div className="space-y-3">
                            <p className="text-xs font-bold text-slate-600 uppercase tracking-widest text-center">Overlay (Original + Prediction)</p>
                            <div className="relative group overflow-hidden rounded-2xl shadow-2xl ring-1 ring-white/10">
                                <img src={`data:image/jpeg;base64,${reportData.overlay}`} className="w-full transition-transform duration-500 group-hover:scale-105" alt="Overlay" />
                            </div>
                          </div>
                          <div className="space-y-3">
                            <p className="text-xs font-bold text-emerald-500/80 uppercase tracking-widest text-center">Ground Truth Mask</p>
                            <img src={`data:image/jpeg;base64,${reportData.ground_truth}`} className="w-full rounded-2xl shadow-2xl ring-1 ring-white/10" alt="GT" />
                          </div>
                        </div>

                        <div className="space-y-6 bg-white/[0.02] p-8 rounded-3xl border border-white/5 shadow-inner">
                          <h3 className="text-lg font-bold text-white">Predicted Class Distribution (Pixel Count)</h3>
                          <div className="space-y-5">
                            {Object.entries(reportData.stats)
                                .sort(([,a]: any, [,b]: any) => b.count - a.count)
                                .filter(([,stat]: any) => stat.count > 0)
                                .map(([cls, stat]: [string, any]) => (
                              <div key={cls} className="space-y-1">
                                <div className="flex justify-between text-xs font-medium">
                                  <span className="text-slate-400">{cls}</span>
                                  <span className="text-slate-600">{stat.count.toLocaleString()} pixels</span>
                                </div>
                                <div className="h-3 w-full bg-white/[0.05] rounded-full overflow-hidden">
                                  <div 
                                    className="h-full transition-all duration-1000 ease-out shadow-[0_0_10px_rgba(255,255,255,0.1)]"
                                    style={{ 
                                      width: `${stat.percentage}%`, 
                                      backgroundColor: CLASS_COLORS[cls] || '#ccc'
                                    }}
                                  />
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="realtime" className="mt-8">
            <RealTimePathFinder />
          </TabsContent>

          <TabsContent value="inference" className="mt-8">
            <Card className="border border-white/5 shadow-2xl bg-slate-900/40 backdrop-blur-xl rounded-2xl overflow-hidden ring-1 ring-white/5">
              <CardHeader className="bg-white/[0.02] border-b border-white/5">
                <CardTitle className="text-white">In-the-Wild Inference</CardTitle>
                <CardDescription className="text-slate-400">
                  Real-time segmentation for field exploration.
                </CardDescription>
              </CardHeader>
              <CardContent className="p-8 space-y-8">
                <div className="grid w-full max-w-sm items-center gap-4">
                  <Label htmlFor="picture" className="text-sm font-bold text-slate-400 uppercase">Input Image</Label>
                  <div className="relative group border-2 border-dashed border-white/10 rounded-2xl p-6 transition-all hover:border-indigo-500/40 hover:bg-white/[0.02]">
                    <Input id="picture" type="file" accept="image/*" onChange={handleSingleFileChange} className="absolute inset-0 opacity-0 cursor-pointer z-10" />
                    <div className="flex flex-col items-center gap-2">
                       {loading ? <Loader2 className="h-8 w-8 animate-spin text-indigo-400" /> : <Upload className="h-8 w-8 text-slate-600" />}
                       <p className="text-sm text-slate-400 font-medium">Click to upload offroad footage</p>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 pt-4">
                  {previewUrl && (
                    <div className="space-y-4">
                      <p className="text-xs font-bold text-slate-600 uppercase tracking-widest text-center">Input Sample</p>
                      <img src={previewUrl} alt="Input Preview" className="w-full rounded-2xl shadow-2xl ring-1 ring-white/10" />
                    </div>
                  )}
                  {resultImages && (
                    <>
                      <div className="space-y-4">
                        <p className="text-xs font-bold text-slate-600 uppercase tracking-widest text-center">Segmentation Mask</p>
                        <img src={resultImages.mask} alt="Segmentation Result" className="w-full rounded-2xl shadow-2xl ring-1 ring-white/10" />
                      </div>
                      <div className="space-y-4">
                        <p className="text-xs font-bold text-indigo-400 uppercase tracking-widest text-center">Path Visualization</p>
                        <img src={resultImages.path} alt="Path Result" className="w-full rounded-2xl shadow-2xl ring-4 ring-indigo-500/20 border border-indigo-500/30" />
                      </div>
                    </>
                  )}
                  {!previewUrl && (
                    <div className="col-span-full py-40 bg-white/[0.01] border-2 border-dashed border-white/[0.03] rounded-3xl flex flex-col items-center justify-center text-slate-700">
                      <ImageIcon className="h-16 w-16 mb-4 opacity-5" />
                      <p className="font-medium">Selected image will appear here</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="iou" className="mt-8">
            <Card className="border border-white/5 shadow-2xl bg-slate-900/40 backdrop-blur-xl rounded-3xl overflow-hidden ring-1 ring-white/5">
              <CardHeader className="bg-white/[0.02] border-b border-white/5">
                <CardTitle className="text-white">Per-Class IoU Analysis</CardTitle>
                <CardDescription className="text-slate-400">Metrics for each segmentation class.</CardDescription>
              </CardHeader>
              <CardContent className="flex justify-center p-12">
                <div className="p-4 bg-white/[0.02] rounded-2xl shadow-2xl border border-white/5 backdrop-blur-md">
                    <img src={getPlotUrl("per_class_iou.png")} alt="Per-Class IoU" className="max-h-[600px] rounded-lg invert brightness-110 contrast-125 hue-rotate-180" />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="matrix" className="mt-8">
            <Card className="border border-white/5 shadow-2xl bg-slate-900/40 backdrop-blur-xl rounded-3xl overflow-hidden ring-1 ring-white/5">
              <CardHeader className="bg-white/[0.02] border-b border-white/5">
                <CardTitle className="text-white">Confusion Matrix</CardTitle>
                <CardDescription className="text-slate-400">Breakdown of true vs predicted classes.</CardDescription>
              </CardHeader>
              <CardContent className="flex justify-center p-12">
                 <div className="p-4 bg-white/[0.02] rounded-2xl shadow-2xl border border-white/5 backdrop-blur-md">
                    <img src={getPlotUrl("confusion_matrix.png")} alt="Confusion Matrix" className="max-h-[600px] rounded-lg invert brightness-110 contrast-125 hue-rotate-180" />
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="dist" className="mt-8">
            <Card className="border border-white/5 shadow-2xl bg-slate-900/40 backdrop-blur-xl rounded-3xl overflow-hidden ring-1 ring-white/5">
              <CardHeader className="bg-white/[0.02] border-b border-white/5">
                <CardTitle className="text-white">Dataset Class Distribution</CardTitle>
                <CardDescription className="text-slate-400">Class representation across the entire dataset.</CardDescription>
              </CardHeader>
              <CardContent className="flex justify-center p-12">
                 <div className="p-4 bg-white/[0.02] rounded-2xl shadow-2xl border border-white/5 backdrop-blur-md">
                    <img src={getPlotUrl("class_distribution.png")} alt="Class Distribution" className="max-h-[600px] rounded-lg invert brightness-110 contrast-125 hue-rotate-180" />
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="metrics" className="mt-8">
            <Card className="border border-white/5 shadow-2xl bg-slate-900/40 backdrop-blur-xl rounded-3xl overflow-hidden ring-1 ring-white/5">
              <CardHeader className="bg-white/[0.02] border-b border-white/5">
                <CardTitle className="text-white">Training Metrics</CardTitle>
                <CardDescription className="text-slate-400">Visualization of training and validation loss/IoU.</CardDescription>
              </CardHeader>
              <CardContent className="flex justify-center p-12">
                 <div className="p-4 bg-white/[0.02] rounded-2xl shadow-2xl border border-white/5 backdrop-blur-md">
                    <img src={getPlotUrl("metrics_plot.png")} alt="Training Metrics" className="max-h-[600px] rounded-lg" />
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        <footer className="text-center text-sm text-slate-500 py-20 border-t border-white/5">
            <div className="flex justify-center gap-8 mb-4">
                <span className="opacity-40">Duality AI Challenge</span>
                <span className="opacity-40">Offroad Semantic Segmentation</span>
                <span className="opacity-40">© 2026</span>
            </div>
            <p className="font-light tracking-widest uppercase text-[10px] text-slate-600">Built for high-performance offroad perception</p>
        </footer>
      </div>
    </div>
  )
}
