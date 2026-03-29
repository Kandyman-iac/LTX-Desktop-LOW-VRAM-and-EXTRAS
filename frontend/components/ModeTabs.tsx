import { cn } from '@/lib/utils'
import { Video, ImageIcon, Scissors, Sparkles, User, Music } from 'lucide-react'

export type GenerationMode = 'text-to-video' | 'image-to-video' | 'text-to-image' | 'retake' | 'ic-lora' | 'magi-human' | 'add-audio'

// Simplified tab modes shown in the UI
type TabMode = 'video' | 'text-to-image' | 'retake' | 'ic-lora' | 'magi-human' | 'add-audio'

interface ModeTabsProps {
  mode: GenerationMode
  onModeChange: (mode: GenerationMode) => void
  disabled?: boolean
  showIcLora?: boolean
}

const mainTabs: { id: TabMode; label: string; genMode: GenerationMode; icon: React.ElementType }[] = [
  { id: 'video', label: 'Video', genMode: 'text-to-video', icon: Video },
  { id: 'text-to-image', label: 'Image', genMode: 'text-to-image', icon: ImageIcon },
  { id: 'retake', label: 'Retake', genMode: 'retake', icon: Scissors },
  { id: 'ic-lora', label: 'IC-LoRA', genMode: 'ic-lora', icon: Sparkles },
]

const experimentalTabs: { id: TabMode; label: string; genMode: GenerationMode; icon: React.ElementType }[] = [
  { id: 'magi-human', label: 'Magi', genMode: 'magi-human', icon: User },
  { id: 'add-audio', label: 'Add Audio', genMode: 'add-audio', icon: Music },
]

export function ModeTabs({ mode, onModeChange, disabled, showIcLora = true }: ModeTabsProps) {
  const activeTab: TabMode = mode === 'text-to-image'
    ? 'text-to-image'
    : mode === 'retake'
      ? 'retake'
      : mode === 'ic-lora'
        ? 'ic-lora'
        : mode === 'magi-human'
          ? 'magi-human'
          : mode === 'add-audio'
            ? 'add-audio'
            : 'video'
  const visibleMainTabs = showIcLora ? mainTabs : mainTabs.filter((tab) => tab.id !== 'ic-lora')

  const renderTab = (tab: typeof mainTabs[number]) => {
    const Icon = tab.icon
    const isActive = activeTab === tab.id
    return (
      <button
        key={tab.id}
        onClick={() => !disabled && onModeChange(tab.genMode)}
        disabled={disabled}
        className={cn(
          'flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
          isActive
            ? 'bg-white text-zinc-900 shadow-sm'
            : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
      >
        <Icon className="h-3.5 w-3.5" />
        {tab.label}
      </button>
    )
  }

  return (
    <div className="flex flex-col gap-1">
      <div className="flex gap-1 p-1 bg-zinc-900 border border-zinc-800 rounded-xl">
        {visibleMainTabs.map(renderTab)}
      </div>
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-semibold tracking-widest text-zinc-600 uppercase pl-1">
          Experimental
        </span>
        <div className="flex gap-1 p-1 bg-zinc-900/60 border border-zinc-800/60 rounded-xl">
          {experimentalTabs.map(renderTab)}
        </div>
      </div>
    </div>
  )
}
