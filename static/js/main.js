/* CodeSage AI - Main JavaScript */

// Utility: Format timestamp
function formatTime(isoString) {
    if (!isoString) return '';
    const d = new Date(isoString);
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
}

// Utility: Truncate text
function truncate(text, maxLen) {
    if (!text) return '';
    return text.length > maxLen ? text.substring(0, maxLen) + '...' : text;
}

// Utility: Format JSON for display
function formatJSON(data) {
    if (typeof data === 'string') {
        try { data = JSON.parse(data); } catch(e) { return data; }
    }
    return JSON.stringify(data, null, 2);
}

// Utility: Confidence color
function confidenceColor(score) {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.5) return 'text-amber-600';
    return 'text-red-600';
}

function confidenceBg(score) {
    if (score >= 0.8) return 'bg-green-100 text-green-700';
    if (score >= 0.5) return 'bg-amber-100 text-amber-700';
    return 'bg-red-100 text-red-700';
}

// Utility: Status badge HTML
function statusBadge(status) {
    const colors = {
        'ready': 'bg-green-100 text-green-700',
        'ingesting': 'bg-yellow-100 text-yellow-700',
        'parsing': 'bg-blue-100 text-blue-700',
        'embedding': 'bg-purple-100 text-purple-700',
        'error': 'bg-red-100 text-red-700',
        'pending': 'bg-slate-100 text-slate-600',
    };
    const cls = colors[status] || colors['pending'];
    return `<span class="px-2 py-0.5 text-xs font-medium rounded-full ${cls}">${status}</span>`;
}

// Utility: Delete project with confirmation
async function deleteProject(projectId, cardElement) {
    if (!confirm('Are you sure you want to delete this project? This cannot be undone.')) return;

    try {
        const resp = await fetch(`/api/project/${projectId}`, { method: 'DELETE' });
        if (resp.ok) {
            if (cardElement) {
                cardElement.style.transition = 'opacity 0.3s, transform 0.3s';
                cardElement.style.opacity = '0';
                cardElement.style.transform = 'scale(0.95)';
                setTimeout(() => cardElement.remove(), 300);
            }
            if (window.showToast) showToast('Project deleted successfully');
        } else {
            if (window.showToast) showToast('Failed to delete project', 'error');
        }
    } catch (e) {
        if (window.showToast) showToast('Error: ' + e.message, 'error');
    }
}
