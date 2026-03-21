import React from 'react'
import { Plus, X, RefreshCw, ChevronLeft, ChevronRight, Layers, GitMerge, FolderPlus, Folder, Trash2, FolderOpen, MonitorPlay } from 'lucide-react'
import type { Asset } from '../../types/project'
import { COLOR_LABELS } from './video-editor-utils'

export interface AssetContextMenuProps {
  asset: Asset
  targetIds: string[]
  assetContextMenu: { assetId: string; x: number; y: number }
  assetContextMenuRef: React.RefObject<HTMLDivElement>
  assets: Asset[]
  bins: string[]
  isRegenerating: boolean
  regeneratingAssetId: string | null
  currentProjectId: string | null
  pushAssetUndoRef: React.RefObject<() => void>
  addClipToTimeline: (asset: Asset, trackIndex?: number, startTime?: number) => void
  loadSourceAsset?: (asset: Asset) => void
  handleRegenerate: (assetId: string) => void
  handleCancelRegeneration: () => void
  setAssetActiveTake: (projectId: string, assetId: string, takeIndex: number) => void
  setTakesViewAssetId: (assetId: string | null) => void
  setSelectedAssetIds: React.Dispatch<React.SetStateAction<Set<string>>>
  setAssetContextMenu: React.Dispatch<React.SetStateAction<{ assetId: string; x: number; y: number } | null>>
  updateAsset: (projectId: string, assetId: string, updates: Partial<Asset>) => void
  addAsset: (projectId: string, asset: Omit<Asset, 'id' | 'createdAt'>) => void
  deleteAsset: (projectId: string, assetId: string) => void
  deleteTakeFromAsset: (projectId: string, assetId: string, takeIndex: number) => void
  setClips: React.Dispatch<React.SetStateAction<import('../../types/project').TimelineClip[]>>
}

export function AssetContextMenu({
  asset,
  targetIds,
  assetContextMenu,
  assetContextMenuRef,
  assets,
  bins,
  isRegenerating,
  regeneratingAssetId,
  currentProjectId,
  pushAssetUndoRef,
  addClipToTimeline,
  loadSourceAsset,
  handleRegenerate,
  handleCancelRegeneration,
  setAssetActiveTake,
  setTakesViewAssetId,
  setSelectedAssetIds,
  setAssetContextMenu,
  updateAsset,
  addAsset,
  deleteAsset,
  deleteTakeFromAsset,
  setClips,
}: AssetContextMenuProps) {
  const isMulti = targetIds.length > 1

  return (
    <div
      ref={assetContextMenuRef}
      className="fixed bg-zinc-800 border border-zinc-700 rounded-xl shadow-2xl py-1.5 z-[60] min-w-[180px] text-xs"
      style={{ left: assetContextMenu.x, top: assetContextMenu.y }}
      onClick={(e) => e.stopPropagation()}
    >
      {isMulti && (
        <div className="px-3 py-1 text-[10px] text-blue-400 font-medium">
          {targetIds.length} assets selected
        </div>
      )}

      {!isMulti && (
        <button
          onClick={() => {
            addClipToTimeline(asset, 0)
            setAssetContextMenu(null)
          }}
          className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
        >
          <Plus className="h-3.5 w-3.5 text-zinc-500" />
          <span>Add to Timeline</span>
        </button>
      )}

      {!isMulti && loadSourceAsset && (asset.type === 'video' || asset.type === 'audio' || asset.type === 'image') && (
        <button
          onClick={() => {
            loadSourceAsset(asset)
            setAssetContextMenu(null)
          }}
          className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
        >
          <MonitorPlay className="h-3.5 w-3.5 text-zinc-500" />
          <span>Open in Clip Viewer</span>
        </button>
      )}

      {!isMulti && asset.path && (
        <button
          onClick={() => {
            window.electronAPI?.showItemInFolder(asset.path!)
            setAssetContextMenu(null)
          }}
          className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
        >
          <FolderOpen className="h-3.5 w-3.5 text-zinc-500" />
          <span>Show in Explorer</span>
        </button>
      )}

      {/* AI regeneration - only for assets with generationParams */}
      {!isMulti && asset.generationParams && (
        <>
          {isRegenerating && regeneratingAssetId === asset.id ? (
            <button
              onClick={() => {
                handleCancelRegeneration()
                setAssetContextMenu(null)
              }}
              className="w-full text-left px-3 py-1.5 text-red-400 hover:bg-zinc-700 flex items-center gap-3"
            >
              <X className="h-3.5 w-3.5" />
              <span>Cancel Regeneration</span>
            </button>
          ) : (
            <button
              onClick={() => {
                handleRegenerate(asset.id)
                setAssetContextMenu(null)
              }}
              disabled={isRegenerating}
              className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3 disabled:opacity-50"
            >
              <RefreshCw className="h-3.5 w-3.5 text-zinc-500" />
              <span>Regenerate</span>
            </button>
          )}
        </>
      )}

      {/* Takes management - for ANY asset with multiple takes */}
      {!isMulti && asset.takes && asset.takes.length > 1 && (
        <>
          <div className="px-3 py-1.5 flex items-center gap-2">
            <span className="text-[10px] text-zinc-500">Take:</span>
            <button
              onClick={(e) => {
                e.stopPropagation()
                if (currentProjectId) {
                  pushAssetUndoRef.current?.()
                  const idx = Math.max(0, (asset.activeTakeIndex ?? 0) - 1)
                  setAssetActiveTake(currentProjectId, asset.id, idx)
                }
              }}
              disabled={(asset.activeTakeIndex ?? 0) === 0}
              className="p-0.5 rounded hover:bg-zinc-600 text-zinc-400 hover:text-white disabled:text-zinc-600 disabled:hover:bg-transparent"
            >
              <ChevronLeft className="h-3 w-3" />
            </button>
            <span className="text-[10px] text-zinc-300 min-w-[28px] text-center">
              {(asset.activeTakeIndex ?? 0) + 1}/{asset.takes.length}
            </span>
            <button
              onClick={(e) => {
                e.stopPropagation()
                if (currentProjectId && asset.takes) {
                  pushAssetUndoRef.current?.()
                  const idx = Math.min(asset.takes.length - 1, (asset.activeTakeIndex ?? 0) + 1)
                  setAssetActiveTake(currentProjectId, asset.id, idx)
                }
              }}
              disabled={asset.takes && (asset.activeTakeIndex ?? 0) >= asset.takes.length - 1}
              className="p-0.5 rounded hover:bg-zinc-600 text-zinc-400 hover:text-white disabled:text-zinc-600 disabled:hover:bg-transparent"
            >
              <ChevronRight className="h-3 w-3" />
            </button>
          </div>
          <button
            onClick={() => {
              setTakesViewAssetId(asset.id)
              setSelectedAssetIds(new Set())
              setAssetContextMenu(null)
            }}
            className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
          >
            <Layers className="h-3.5 w-3.5 text-zinc-500" />
            <span>View All Takes</span>
          </button>
          <button
            onClick={() => {
              if (!currentProjectId || !asset.takes) return
              pushAssetUndoRef.current?.()
              asset.takes.slice(1).forEach(take => {
                addAsset(currentProjectId, {
                  type: asset.type,
                  path: take.path,
                  url: take.url,
                  prompt: asset.prompt,
                  resolution: asset.resolution,
                  duration: asset.duration,
                  thumbnail: take.thumbnail,
                  generationParams: asset.generationParams,
                  takes: [{ url: take.url, path: take.path, thumbnail: take.thumbnail, createdAt: take.createdAt }],
                  activeTakeIndex: 0,
                })
              })
              const firstTake = asset.takes[0]
              updateAsset(currentProjectId, asset.id, {
                takes: [firstTake],
                activeTakeIndex: 0,
                url: firstTake.url,
                path: firstTake.path,
                thumbnail: firstTake.thumbnail || asset.thumbnail,
              })
              setAssetContextMenu(null)
            }}
            className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
          >
            <GitMerge className="h-3.5 w-3.5 text-zinc-500 rotate-180" />
            <span>Ungroup Takes</span>
          </button>
          <button
            onClick={() => {
              const activeIdx = asset.activeTakeIndex ?? 0
              if (confirm(`Delete take ${activeIdx + 1}?`)) {
                if (currentProjectId && asset.takes) {
                  pushAssetUndoRef.current?.()
                  setClips(prev => prev.map(c => {
                    if (c.assetId !== asset.id) return c
                    const cIdx = c.takeIndex ?? (asset.activeTakeIndex ?? asset.takes!.length - 1)
                    if (cIdx === activeIdx) {
                      return { ...c, takeIndex: Math.max(0, activeIdx - 1) }
                    } else if (cIdx > activeIdx) {
                      return { ...c, takeIndex: cIdx - 1 }
                    }
                    return c
                  }))
                  deleteTakeFromAsset(currentProjectId, asset.id, activeIdx)
                }
              }
              setAssetContextMenu(null)
            }}
            className="w-full text-left px-3 py-1.5 text-red-400 hover:bg-red-900/30 flex items-center gap-3"
          >
            <Trash2 className="h-3.5 w-3.5" />
            <span>Delete Active Take</span>
          </button>
        </>
      )}

      <div className="h-px bg-zinc-700 my-1" />

      {/* Color label picker */}
      <div className="px-3 py-1 text-[10px] text-zinc-500 font-semibold uppercase tracking-wider">Label</div>
      <div className="px-3 py-1.5 flex items-center gap-1 flex-wrap">
        {/* Clear color button */}
        <button
          onClick={() => {
            if (currentProjectId) {
              pushAssetUndoRef.current?.()
              targetIds.forEach(id => updateAsset(currentProjectId, id, { colorLabel: undefined }))
              // Sync: also clear colorLabel on all timeline clips referencing these assets
              const ids = new Set(targetIds)
              setClips(prev => prev.map(c => c.assetId && ids.has(c.assetId) ? { ...c, colorLabel: undefined } : c))
            }
            setAssetContextMenu(null)
          }}
          className={`w-4 h-4 rounded-full border-2 flex items-center justify-center transition-all ${
            !asset.colorLabel ? 'border-white scale-110' : 'border-zinc-600 hover:border-zinc-400'
          }`}
          title="No label"
        >
          <X className="h-2 w-2 text-zinc-400" />
        </button>
        {COLOR_LABELS.map(cl => (
          <button
            key={cl.id}
            onClick={() => {
              if (currentProjectId) {
                pushAssetUndoRef.current?.()
                targetIds.forEach(id => updateAsset(currentProjectId, id, { colorLabel: cl.id }))
                // Sync: also set colorLabel on all timeline clips referencing these assets
                const ids = new Set(targetIds)
                setClips(prev => prev.map(c => c.assetId && ids.has(c.assetId) ? { ...c, colorLabel: cl.id } : c))
              }
              setAssetContextMenu(null)
            }}
            className={`w-4 h-4 rounded-full transition-all ${
              asset.colorLabel === cl.id ? 'ring-2 ring-white ring-offset-1 ring-offset-zinc-800 scale-110' : 'hover:scale-125'
            }`}
            style={{ backgroundColor: cl.color }}
            title={cl.label}
          />
        ))}
      </div>

      <div className="h-px bg-zinc-700 my-1" />

      <div className="px-3 py-1 text-[10px] text-zinc-500 font-semibold uppercase tracking-wider">Move to Bin</div>

      <button
        onClick={() => {
          if (currentProjectId) {
            pushAssetUndoRef.current?.()
            targetIds.forEach(id => updateAsset(currentProjectId, id, { bin: undefined }))
          }
          setAssetContextMenu(null)
          setSelectedAssetIds(new Set())
        }}
        className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
      >
        <X className="h-3.5 w-3.5 text-zinc-500" />
        <span>Remove from Bin</span>
      </button>

      {bins.map(bin => (
        <button
          key={bin}
          onClick={() => {
            if (currentProjectId) {
              pushAssetUndoRef.current?.()
              targetIds.forEach(id => updateAsset(currentProjectId, id, { bin }))
            }
            setAssetContextMenu(null)
            setSelectedAssetIds(new Set())
          }}
          className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
        >
          <Folder className="h-3.5 w-3.5 text-zinc-500" />
          <span>{bin}</span>
        </button>
      ))}

      <button
        onClick={() => {
          const name = prompt('New bin name:')
          if (name?.trim() && currentProjectId) {
            pushAssetUndoRef.current?.()
            targetIds.forEach(id => updateAsset(currentProjectId, id, { bin: name.trim() }))
          }
          setAssetContextMenu(null)
          setSelectedAssetIds(new Set())
        }}
        className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
      >
        <FolderPlus className="h-3.5 w-3.5 text-zinc-500" />
        <span>New Bin...</span>
      </button>

      {isMulti && (
        <>
          <div className="h-px bg-zinc-700 my-1" />
          <button
            onClick={() => {
              if (!currentProjectId) return
              const selectedAssets = assets.filter(a => targetIds.includes(a.id))
              if (selectedAssets.length < 2) return
              pushAssetUndoRef.current?.()
              const primary = selectedAssets[0]
              const newTakes = selectedAssets.map(a => ({
                url: a.url, path: a.path, thumbnail: a.thumbnail, createdAt: a.createdAt,
              }))
              updateAsset(currentProjectId, primary.id, { takes: newTakes, activeTakeIndex: 0 })
              selectedAssets.slice(1).forEach(a => deleteAsset(currentProjectId, a.id))
              setSelectedAssetIds(new Set())
              setAssetContextMenu(null)
            }}
            className="w-full text-left px-3 py-1.5 text-blue-300 hover:bg-zinc-700 flex items-center gap-3"
          >
            <GitMerge className="h-3.5 w-3.5" />
            <span>Group as Takes</span>
          </button>
          <button
            onClick={() => {
              setSelectedAssetIds(new Set())
              setAssetContextMenu(null)
            }}
            className="w-full text-left px-3 py-1.5 text-zinc-300 hover:bg-zinc-700 flex items-center gap-3"
          >
            <X className="h-3.5 w-3.5 text-zinc-500" />
            <span>Clear Selection</span>
          </button>
        </>
      )}

      <div className="h-px bg-zinc-700 my-1" />

      <button
        onClick={() => {
          if (currentProjectId) {
            pushAssetUndoRef.current?.()
            targetIds.forEach(id => deleteAsset(currentProjectId, id))
          }
          setAssetContextMenu(null)
          setSelectedAssetIds(new Set())
        }}
        className="w-full text-left px-3 py-1.5 text-red-400 hover:bg-zinc-700 flex items-center gap-3"
      >
        <Trash2 className="h-3.5 w-3.5" />
        <span>{isMulti ? `Delete ${targetIds.length} Assets` : 'Delete Asset'}</span>
      </button>
    </div>
  )
}
