import { GlassTimeCard } from "@/components/ui/glass-time-card"

export function Demo() {
  return (
    <div className="w-screen h-screen bg-cover bg-center flex items-center justify-center" style={{ backgroundImage: "url('https://images.unsplash.com/photo-1533158326339-7f3cf2404354?q=80&w=1068&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')" }}>
        <GlassTimeCard showSeconds showTimezone />
    </div>
  )
}
