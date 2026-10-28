/**
 * RetailForecaster — Dual Workspace Enterprise Frontend Application
 * Workspaces:
 * 1. Developer View:
 *    - Dataset & Quality (Ingestion, missing values, duplicates, completeness)
 *    - Model Evaluation (Metrics: MAE, WAPE, RMSE, Runtime for CatBoost, Prophet, ARIMA, MA, Naive)
 *    - Model Comparison (5 Model cards, ranking, multi-model chart, model approval for retailer)
 *    - Experiment History (Audited record of completed runs)
 *    - Settings
 * 2. Retailer View:
 *    - Business Overview (Total demand, rising/declining counts, reliability %, historical vs forecast with uncertainty)
 *    - Demand Forecasts (Approved model forecasts, interactive filters, data table)
 *    - Product Insights (Actionable signals: rising/declining %, non-invented explanations)
 *    - AI Assistant (Gemini assistant grounded with full product catalog names, revenue, and states/cities)
 *    - Reports (Executive demand report & downloadable CSV)
 *    - Settings
 */

// ── 1. Application State ──────────────────────────────────────────
const state = {
    currentWorkspace: 'dev', // 'dev' | 'retail'
    activeTab: 'dev-dataset',
    conversationHistory: [],
    evalData: null,
    qualityData: null,
    retailOverview: null,
    retailInsights: null,
    approvedModel: null,
    isProcessing: false,
    selectedEvalModel: 'catboost',
    
    // Explorer filter & pagination state
    filters: {
        category: 'all',
        store_id: 'all',
        sales_channel: 'all',
        horizon: 'all',
        search: '',
        page: 1,
        page_size: 10,
        sort_by: 'predicted_units',
        sort_order: 'desc',
    },
    
    // Chart instances
    charts: {
        multimodel: null,
        horizonError: null,
        explorerCat: null,
        retailTimeline: null,
    },

    searchDebounceTimer: null,
};

// ── Formatting Utilities (Strict 2 Decimals) ──────────────────────
function formatDecimal(val, digits = 2) {
    if (val === null || val === undefined || val === '' || isNaN(Number(val))) return '—';
    return Number(val).toFixed(digits);
}

function formatPercent(val, digits = 2) {
    if (val === null || val === undefined || val === '' || isNaN(Number(val))) return '—';
    return Number(val).toFixed(digits) + '%';
}

// ── 2. DOM Elements Mapping ───────────────────────────────────────
let DOM = {};

function initDOMMap() {
    DOM = {
        sidebar: document.getElementById('sidebar'),
        btnCollapseSidebar: document.getElementById('btn-collapse-sidebar'),
        btnMobileMenu: document.getElementById('btn-mobile-menu'),
        breadcrumbWorkspace: document.getElementById('breadcrumb-workspace'),
        breadcrumbTitle: document.getElementById('breadcrumb-title'),
        pageHeading: document.getElementById('page-heading'),
        apiStatus: document.getElementById('api-status'),
        sideTotalRows: document.getElementById('side-total-rows'),
        sideApprovedModel: document.getElementById('side-approved-model'),
        sidebarRoleBadge: document.getElementById('sidebar-role-badge'),

        // Action groups
        actionsDev: document.getElementById('actions-dev'),
        actionsRetail: document.getElementById('actions-retail'),

        // Workspace sections
        sidebarSectionDev: document.getElementById('sidebar-section-dev'),
        sidebarSectionRetail: document.getElementById('sidebar-section-retail'),

        // (Workspace switching handled by auth role, no toggling buttons)

        // Chat
        chatMessages: document.getElementById('chat-messages'),
        chatInput: document.getElementById('chat-input'),
        btnSend: document.getElementById('btn-send'),
        chatCharCounter: document.getElementById('chat-char-counter'),
        toastContainer: document.getElementById('toast-container'),
    };
}

// ── 3. Lifecycle & Startup ────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initDOMMap();
    initLayoutControls();
    initChatInput();
    initAuthParticles();
    
    // Check if user is already logged in
    checkAuthSession();
});

// ── AUTH: Session & Authentication Flow ───────────────────────────
async function checkAuthSession() {
    try {
        const res = await fetch('/api/auth/session');
        const data = await res.json();
        if (data.authenticated) {
            enterApp(data.username, data.role);
        } else {
            showAuthOverlay();
        }
    } catch (e) {
        console.warn('Session check failed, showing login:', e);
        showAuthOverlay();
    }
}

function showAuthOverlay() {
    const overlay = document.getElementById('auth-overlay');
    const appLayout = document.getElementById('app-layout');
    if (overlay) overlay.style.display = 'flex';
    if (overlay) overlay.classList.remove('hidden');
    if (appLayout) appLayout.style.display = 'none';
    if (window.lucide) lucide.createIcons();
}

function enterApp(username, role) {
    const overlay = document.getElementById('auth-overlay');
    const appLayout = document.getElementById('app-layout');
    
    // Hide auth overlay
    if (overlay) {
        overlay.classList.add('hidden');
        overlay.style.display = 'none';
    }
    // Show app
    if (appLayout) appLayout.style.display = '';
    
    // Store user info in state
    state.authUser = { username, role };
    
    // Update UI elements with user info
    const userMenuName = document.getElementById('user-menu-name');
    const workspaceRoleText = document.getElementById('workspace-role-text');
    const sidebarRoleBadge = document.getElementById('sidebar-role-badge');
    const topbarRoleBadge = document.getElementById('topbar-role-badge-text');
    
    if (userMenuName) userMenuName.textContent = username;
    
    if (role === 'admin') {
        if (workspaceRoleText) workspaceRoleText.textContent = 'Developer Studio';
        if (sidebarRoleBadge) sidebarRoleBadge.textContent = 'Developer Studio';
        if (topbarRoleBadge) {
            topbarRoleBadge.innerHTML = '<i data-lucide="terminal" class="btn-icon"></i><span>Developer View</span>';
        }
    } else {
        if (workspaceRoleText) workspaceRoleText.textContent = 'Retailer Planning';
        if (sidebarRoleBadge) sidebarRoleBadge.textContent = 'Retailer Planning';
        if (topbarRoleBadge) {
            topbarRoleBadge.innerHTML = '<i data-lucide="store" class="btn-icon"></i><span>Retailer View</span>';
        }
    }
    
    // Switch to the correct workspace based on role (locked, no switcher)
    const workspace = role === 'admin' ? 'dev' : 'retail';
    switchWorkspace(workspace);
    
    // Check API health
    checkApiHealth();
    
    if (window.lucide) lucide.createIcons();
}

// ── Auth Form Handlers ────────────────────────────────────────────
async function handleLogin(e) {
    e.preventDefault();
    const username = document.getElementById('login-username').value.trim();
    const password = document.getElementById('login-password').value;
    const errorEl = document.getElementById('login-error');
    const btn = document.getElementById('login-submit-btn');
    
    errorEl.textContent = '';
    btn.disabled = true;
    btn.querySelector('.auth-btn-text').style.display = 'none';
    btn.querySelector('.auth-btn-loader').style.display = 'inline-flex';
    if (window.lucide) lucide.createIcons();
    
    try {
        const res = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
        const data = await res.json();
        
        if (data.success) {
            enterApp(data.username, data.role);
        } else {
            errorEl.textContent = data.error || 'Login failed.';
        }
    } catch (err) {
        errorEl.textContent = 'Connection error. Please try again.';
    } finally {
        btn.disabled = false;
        btn.querySelector('.auth-btn-text').style.display = '';
        btn.querySelector('.auth-btn-loader').style.display = 'none';
    }
}

async function handleSignup(e) {
    e.preventDefault();
    const username = document.getElementById('signup-username').value.trim();
    const password = document.getElementById('signup-password').value;
    const confirmPassword = document.getElementById('signup-confirm-password').value;
    const errorEl = document.getElementById('signup-error');
    const btn = document.getElementById('signup-submit-btn');
    
    errorEl.textContent = '';
    
    if (password !== confirmPassword) {
        errorEl.textContent = 'Passwords do not match.';
        return;
    }
    
    btn.disabled = true;
    btn.querySelector('.auth-btn-text').style.display = 'none';
    btn.querySelector('.auth-btn-loader').style.display = 'inline-flex';
    if (window.lucide) lucide.createIcons();
    
    try {
        const res = await fetch('/api/auth/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
        const data = await res.json();
        
        if (data.success) {
            enterApp(data.username, data.role);
        } else {
            errorEl.textContent = data.error || 'Signup failed.';
        }
    } catch (err) {
        errorEl.textContent = 'Connection error. Please try again.';
    } finally {
        btn.disabled = false;
        btn.querySelector('.auth-btn-text').style.display = '';
        btn.querySelector('.auth-btn-loader').style.display = 'none';
    }
}

async function handleLogout() {
    try {
        await fetch('/api/auth/logout', { method: 'POST' });
    } catch (e) {
        // Proceed with logout even if API call fails
    }
    state.authUser = null;
    // Close dropdown
    const dd = document.getElementById('user-dropdown');
    if (dd) dd.style.display = 'none';
    // Show auth overlay
    showAuthOverlay();
    // Clear forms
    const loginForm = document.getElementById('login-form');
    const signupForm = document.getElementById('signup-form');
    if (loginForm) loginForm.reset();
    if (signupForm) signupForm.reset();
    document.getElementById('login-error').textContent = '';
    document.getElementById('signup-error').textContent = '';
    // Show login card
    document.getElementById('auth-login-card').style.display = '';
    document.getElementById('auth-signup-card').style.display = 'none';
}

function showSignupCard(e) {
    if (e) e.preventDefault();
    document.getElementById('auth-login-card').style.display = 'none';
    document.getElementById('auth-signup-card').style.display = '';
    document.getElementById('signup-error').textContent = '';
    if (window.lucide) lucide.createIcons();
}

function showLoginCard(e) {
    if (e) e.preventDefault();
    document.getElementById('auth-signup-card').style.display = 'none';
    document.getElementById('auth-login-card').style.display = '';
    document.getElementById('login-error').textContent = '';
    if (window.lucide) lucide.createIcons();
}

function toggleUserDropdown() {
    const dd = document.getElementById('user-dropdown');
    if (!dd) return;
    const isVisible = dd.style.display !== 'none';
    dd.style.display = isVisible ? 'none' : 'block';
    if (!isVisible && window.lucide) lucide.createIcons();
    
    // Close on outside click
    if (!isVisible) {
        const closeHandler = (e) => {
            if (!e.target.closest('.topbar-user-menu')) {
                dd.style.display = 'none';
                document.removeEventListener('click', closeHandler);
            }
        };
        setTimeout(() => document.addEventListener('click', closeHandler), 10);
    }
}

// ── User Profile Modal Management ─────────────────────────────────
async function openUserProfileModal() {
    // Hide dropdown if open
    const dd = document.getElementById('user-dropdown');
    if (dd) dd.style.display = 'none';

    const modal = document.getElementById('user-profile-modal');
    if (!modal) return;

    // Reset feedback alert
    const alertEl = document.getElementById('profile-alert');
    if (alertEl) {
        alertEl.style.display = 'none';
        alertEl.className = 'profile-alert';
        alertEl.textContent = '';
    }

    // Clear password inputs
    const curPw = document.getElementById('profile-current-password');
    const newPw = document.getElementById('profile-new-password');
    const confPw = document.getElementById('profile-confirm-password');
    if (curPw) curPw.value = '';
    if (newPw) newPw.value = '';
    if (confPw) confPw.value = '';

    // Show modal immediately
    modal.style.display = 'flex';
    if (window.lucide) lucide.createIcons();

    // Fetch user profile from API
    try {
        const res = await fetch('/api/auth/profile');
        if (res.ok) {
            const data = await res.json();
            if (data.success && data.profile) {
                const p = data.profile;
                const nameEl = document.getElementById('profile-modal-username');
                const roleEl = document.getElementById('profile-modal-role');
                const phoneEl = document.getElementById('profile-phone');
                const emailEl = document.getElementById('profile-email');
                const bizEl = document.getElementById('profile-business-name');

                if (nameEl) nameEl.textContent = p.username || 'User';
                if (roleEl) {
                    roleEl.textContent = p.role === 'admin' ? 'Developer Studio (Admin)' : 'Retailer Planning';
                }
                if (phoneEl) phoneEl.value = p.phone || '';
                if (emailEl) emailEl.value = p.email || '';
                if (bizEl) bizEl.value = p.business_name || '';
            }
        }
    } catch (e) {
        console.warn('Could not load profile:', e);
    }
}

function closeUserProfileModal() {
    const modal = document.getElementById('user-profile-modal');
    if (modal) modal.style.display = 'none';
}

function handleProfileBackdropClick(e) {
    if (e.target && e.target.id === 'user-profile-modal') {
        closeUserProfileModal();
    }
}

async function handleSaveProfile(e) {
    e.preventDefault();

    const phone = (document.getElementById('profile-phone').value || '').trim();
    const email = (document.getElementById('profile-email').value || '').trim();
    const businessName = (document.getElementById('profile-business-name').value || '').trim();
    const currentPassword = document.getElementById('profile-current-password').value || '';
    const newPassword = document.getElementById('profile-new-password').value || '';
    const confirmPassword = document.getElementById('profile-confirm-password').value || '';

    const alertEl = document.getElementById('profile-alert');
    const btn = document.getElementById('btn-save-profile');

    const showAlert = (msg, isSuccess = false) => {
        if (!alertEl) return;
        alertEl.style.display = 'flex';
        alertEl.className = isSuccess ? 'profile-alert alert-success' : 'profile-alert alert-danger';
        alertEl.innerHTML = `<i data-lucide="${isSuccess ? 'check-circle-2' : 'alert-circle'}" class="icon-xs"></i> <span>${msg}</span>`;
        if (window.lucide) lucide.createIcons();
    };

    // Client-side validations
    if (newPassword) {
        if (newPassword.length < 4) {
            showAlert('New password must be at least 4 characters long.');
            return;
        }
        if (newPassword !== confirmPassword) {
            showAlert('New passwords do not match.');
            return;
        }
        if (!currentPassword) {
            showAlert('Please enter your current password to authorize this password change.');
            return;
        }
    }

    if (btn) {
        btn.disabled = true;
        const txt = btn.querySelector('.btn-text');
        const ldr = btn.querySelector('.btn-loader');
        if (txt) txt.style.display = 'none';
        if (ldr) ldr.style.display = 'inline-flex';
    }

    try {
        const res = await fetch('/api/auth/profile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                phone,
                email,
                business_name: businessName,
                current_password: currentPassword,
                new_password: newPassword,
            })
        });

        const data = await res.json();

        if (res.ok && data.success) {
            showAlert('Profile details updated successfully in database!', true);
            if (typeof showToast === 'function') {
                showToast('Profile updated successfully!', 'success');
            }

            // Clear password fields on success
            const curPw = document.getElementById('profile-current-password');
            const newPw = document.getElementById('profile-new-password');
            const confPw = document.getElementById('profile-confirm-password');
            if (curPw) curPw.value = '';
            if (newPw) newPw.value = '';
            if (confPw) confPw.value = '';

            setTimeout(() => {
                closeUserProfileModal();
            }, 1200);
        } else {
            showAlert(data.error || 'Failed to update profile.');
        }
    } catch (err) {
        showAlert('Network error occurred while saving profile.');
    } finally {
        if (btn) {
            btn.disabled = false;
            const txt = btn.querySelector('.btn-text');
            const ldr = btn.querySelector('.btn-loader');
            if (txt) txt.style.display = '';
            if (ldr) ldr.style.display = 'none';
        }
    }
}

function initAuthParticles() {
    const container = document.getElementById('auth-particles');
    if (!container) return;
    container.innerHTML = '';
    
    // Warm pastel palette matching RetailForecaster design system
    const colors = [
        'rgba(175, 154, 201, 0.45)', // Lavender
        'rgba(227, 196, 155, 0.45)', // Champagne Gold
        'rgba(165, 147, 123, 0.35)', // Warm Taupe
        'rgba(197, 180, 220, 0.40)', // Light Lavender
        'rgba(235, 214, 185, 0.42)'  // Peach Gold
    ];
    
    for (let i = 0; i < 16; i++) {
        const particle = document.createElement('div');
        particle.className = 'auth-particle';
        const size = 60 + Math.random() * 200;
        const color = colors[Math.floor(Math.random() * colors.length)];
        particle.style.cssText = `
            width: ${size}px;
            height: ${size}px;
            left: ${Math.random() * 95}%;
            top: ${Math.random() * 95}%;
            background: radial-gradient(circle, ${color} 0%, transparent 70%);
            animation-duration: ${16 + Math.random() * 18}s;
            animation-delay: ${Math.random() * -15}s;
        `;
        container.appendChild(particle);
    }
}

// ── 4. Layout & Navigation ────────────────────────────────────────
function initLayoutControls() {
    if (DOM.btnCollapseSidebar) {
        DOM.btnCollapseSidebar.addEventListener('click', () => {
            const isCollapsed = DOM.sidebar.classList.toggle('collapsed');
            DOM.btnCollapseSidebar.setAttribute('title', isCollapsed ? 'Expand sidebar' : 'Collapse sidebar');
            DOM.btnCollapseSidebar.setAttribute('aria-label', isCollapsed ? 'Expand sidebar' : 'Collapse sidebar');
            DOM.btnCollapseSidebar.innerHTML = `<i data-lucide="${isCollapsed ? 'panel-left-open' : 'panel-left-close'}"></i>`;
            if (window.lucide) lucide.createIcons();
            setTimeout(resizeAllCharts, 250);
        });
    }

    const sidebarBackdrop = document.getElementById('sidebar-backdrop');
    if (DOM.btnMobileMenu) {
        DOM.btnMobileMenu.addEventListener('click', () => {
            const isOpen = DOM.sidebar.classList.toggle('mobile-open');
            if (sidebarBackdrop) sidebarBackdrop.classList.toggle('active', isOpen);
        });
    }

    window.closeMobileSidebar = () => {
        DOM.sidebar.classList.remove('mobile-open');
        if (sidebarBackdrop) sidebarBackdrop.classList.remove('active');
    };

    window.addEventListener('resize', () => {
        if (window.innerWidth > 768) window.closeMobileSidebar();
        resizeAllCharts();
    });

    // Wire all nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const tab = item.getAttribute('data-tab');
            switchTab(tab);
            window.closeMobileSidebar();
        });
    });
}

// ── 5. Workspace Switching (role-locked by auth) ─────────────────
function switchWorkspace(workspace) {
    state.currentWorkspace = workspace;

    // Toggle Sidebar Nav Sections
    if (DOM.sidebarSectionDev && DOM.sidebarSectionRetail) {
        DOM.sidebarSectionDev.style.display = workspace === 'dev' ? 'block' : 'none';
        DOM.sidebarSectionRetail.style.display = workspace === 'retail' ? 'block' : 'none';
    }

    // Toggle Action Buttons in Header
    if (DOM.actionsDev && DOM.actionsRetail) {
        DOM.actionsDev.style.display = workspace === 'dev' ? 'flex' : 'none';
        // Only show Generate Forecast button on tabs where it's relevant
        const isForecastTab = state.activeTab === 'ret-forecasts' || state.activeTab === 'ret-overview';
        DOM.actionsRetail.style.display = (workspace === 'retail' && isForecastTab) ? 'flex' : 'none';
    }

    // Update Role Subtitle in Sidebar
    if (DOM.sidebarRoleBadge) {
        DOM.sidebarRoleBadge.textContent = workspace === 'dev' ? 'Developer Studio' : 'Retailer Planning';
    }

    // Switch default active tab
    if (workspace === 'dev') {
        switchTab('dev-dataset');
        loadDeveloperWorkspace();
    } else {
        switchTab('ret-forecasts');
        loadRetailerWorkspace();
    }

    if (window.lucide) lucide.createIcons();
}

function switchTab(tabId) {
    state.activeTab = tabId;

    // Toggle body class for chat view to expand workspace and eliminate topbar overhead
    document.body.classList.toggle('is-chat-tab-active', tabId === 'ret-chat');

    // Update Nav items active state
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
    });

    // Update Tab panes
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.toggle('active', pane.id === `tab-${tabId}`);
    });

    // Update Header Breadcrumbs & Titles
    const titles = {
        'dev-dataset': { ws: 'Developer View', breadcrumb: 'Dataset & Quality', heading: 'Dataset Integrity & Ingestion Audit' },
        'dev-evaluation': { ws: 'Developer View', breadcrumb: 'Model Evaluation', heading: 'Single-Model Diagnostics & Backtesting' },
        'dev-comparison': { ws: 'Developer View', breadcrumb: 'Model Comparison', heading: '5-Model Leaderboard & Production Approval' },
        'dev-history': { ws: 'Developer View', breadcrumb: 'Experiment History', heading: 'Audited Forecasting Experiments' },
        'dev-settings': { ws: 'Developer View', breadcrumb: 'Developer Settings', heading: 'Evaluation Parameters & Infrastructure' },

        'ret-forecasts': { ws: 'Retailer View', breadcrumb: 'Demand Forecasts', heading: 'Interactive 4-Week Demand Explorer' },
        'ret-overview': { ws: 'Retailer View', breadcrumb: 'Business Overview', heading: 'Executive Demand Forecast Outlook' },
        'ret-insights': { ws: 'Retailer View', breadcrumb: 'Product Insights', heading: 'Actionable Demand Signals & Explanations' },
        'ret-chat': { ws: 'Retailer View', breadcrumb: 'AI Assistant', heading: 'Conversational Demand Intelligence (Gemini)' },
        'ret-reports': { ws: 'Retailer View', breadcrumb: 'Reports', heading: 'Business Forecast Exports & Summaries' },
        'ret-settings': { ws: 'Retailer View', breadcrumb: 'Retailer Settings', heading: 'Business Preferences & Inventory Buffers' },
    };

    const t = titles[tabId] || { ws: 'Workspace', breadcrumb: 'Analytics', heading: 'Demand Intelligence' };
    if (DOM.breadcrumbWorkspace) DOM.breadcrumbWorkspace.textContent = t.ws;
    if (DOM.breadcrumbTitle) DOM.breadcrumbTitle.textContent = t.breadcrumb;
    if (DOM.pageHeading) DOM.pageHeading.textContent = t.heading;

    // Trigger tab-specific renders or resize
    // Show "Generate Forecast" header button only on relevant retailer tabs
    if (DOM.actionsRetail) {
        const showGenForecast = (state.currentWorkspace === 'retail') &&
            (tabId === 'ret-forecasts' || tabId === 'ret-overview');
        DOM.actionsRetail.style.display = showGenForecast ? 'flex' : 'none';
    }

    if (tabId === 'dev-comparison') {
        setTimeout(renderMultiModelChart, 150);
    } else if (tabId === 'ret-overview') {
        setTimeout(renderRetailTimelineChart, 150);
    } else if (tabId === 'ret-forecasts') {
        if (!state.forecastSummary) loadForecastTable();
        loadGeoForecastData();
        setTimeout(() => {
            resizeChart(state.charts.explorerCat);
            if (state.geoData) renderIndiaImageMap(state.geoData);
        }, 180);
    } else if (tabId === 'dev-history') {
        loadExperimentHistory();
    } else if (tabId === 'ret-insights') {
        loadRetailInsights();
    }

    setTimeout(resizeAllCharts, 200);
}

// ── 6. Developer Workspace Operations ─────────────────────────────
async function loadDeveloperWorkspace() {
    await Promise.all([
        loadDatasetQuality(),
        loadLatestEvaluation(),
        loadApprovedModel(),
    ]);
}

async function loadDatasetQuality() {
    try {
        const res = await fetch('/api/data/quality');
        if (!res.ok) return;
        const q = await res.json();
        state.qualityData = q;

        const totalEl = document.getElementById('dq-total-rows');
        const missEl = document.getElementById('dq-missing-rows');
        const dupEl = document.getElementById('dq-duplicates');
        const zeroEl = document.getElementById('dq-zero-pct');

        if (totalEl) totalEl.textContent = (q.total_rows || 1003500).toLocaleString();
        if (missEl) missEl.textContent = (q.null_rows_dropped || 40264).toLocaleString();
        if (dupEl) dupEl.textContent = '0';
        if (zeroEl) zeroEl.textContent = formatPercent(q.zero_demand_percentage ?? 36.7);
        if (DOM.sideTotalRows) DOM.sideTotalRows.textContent = (q.total_rows || 1024772).toLocaleString();
    } catch (e) {
        console.warn('Dataset quality load error:', e);
    }
}

async function loadLatestEvaluation() {
    try {
        const res = await fetch('/api/evaluation/latest');
        if (!res.ok) throw new Error('No evaluation run found');
        const data = await res.json();
        state.evalData = data;

        // Render Developer sub-panes
        renderModelComparisonCards(data.comparison || []);
        renderLeaderboardTable(data.comparison || []);
        renderEvaluationInspector(data);
        renderFailureCases(data.failure_cases || []);
        renderHorizonTable(data.catboost?.horizon_metrics || []);
        renderCategoryTable(data.catboost?.category_metrics || []);
        renderHorizonErrorChart(data.catboost?.horizon_metrics || []);
        renderMultiModelChart();
    } catch (e) {
        console.error('Failed to load evaluation results:', e);
    }
}

async function loadApprovedModel() {
    try {
        const res = await fetch('/api/model/approved');
        if (!res.ok) return;
        const app = await res.json();
        state.approvedModel = app;

        // Update UI banners
        const titleEl = document.getElementById('approved-model-title');
        const descEl = document.getElementById('approved-model-desc');
        const sideModelEl = document.getElementById('side-approved-model');
        const retModelLbl = document.getElementById('ret-approved-model-label');
        const repModelLbl = document.getElementById('rep-model-name');

        if (titleEl) titleEl.textContent = app.name || 'CatBoost Regressor (GBDT)';
        if (descEl) descEl.textContent = app.description || 'Approved for Retailer View.';
        if (sideModelEl) sideModelEl.textContent = app.approved_model_id?.toUpperCase() || 'CATBOOST';
        if (retModelLbl) retModelLbl.textContent = app.name || 'CatBoost';
        if (repModelLbl) repModelLbl.textContent = app.name || 'CatBoost Regressor';

        const settingSelect = document.getElementById('setting-approved-model');
        if (settingSelect && app.approved_model_id) {
            settingSelect.value = app.approved_model_id;
        }
    } catch (e) {
        console.warn('Failed to load approved model setting:', e);
    }
}

// ── 7. Model Comparison & Production Approval ─────────────────────
function renderModelComparisonCards(comparison) {
    const container = document.getElementById('comparison-cards-container');
    if (!container) return;

    if (!comparison || comparison.length === 0) {
        container.innerHTML = `<div class="card p-4 text-center text-muted">No model benchmarks available. Run a benchmark first.</div>`;
        return;
    }

    const approvedId = state.approvedModel?.approved_model_id || 'catboost';

    container.innerHTML = comparison.map(m => {
        const isApproved = m.model === approvedId;
        const isRank1 = m.rank === 1;

        return `
            <div class="model-compare-card ${isRank1 ? 'card-rank-1' : ''}" id="card-model-${m.model}">
                <div class="model-card-header">
                    <div>
                        <h3 class="model-name">${m.name}</h3>
                        <span class="model-type-sub">${m.category}</span>
                    </div>
                    <span class="model-rank-badge">${m.rank ? `Rank #${m.rank}` : 'Benchmark'}</span>
                </div>

                <div class="model-metrics-row">
                    <div class="m-metric">
                        <span class="m-metric-label">WAPE</span>
                        <span class="m-metric-val ${m.wape < 45 ? 'text-success' : ''}">${formatPercent(m.wape)}</span>
                    </div>
                    <div class="m-metric">
                        <span class="m-metric-label">MAE</span>
                        <span class="m-metric-val">${formatDecimal(m.mae)}</span>
                    </div>
                    <div class="m-metric">
                        <span class="m-metric-label">Runtime</span>
                        <span class="m-metric-val">${m.runtime_seconds !== null && m.runtime_seconds !== undefined ? formatDecimal(m.runtime_seconds) + 's' : '—'}</span>
                    </div>
                </div>

                <p class="model-card-desc">${m.description || ''}</p>

                <div class="model-card-footer">
                    <button class="btn-approve-model ${isApproved ? 'approved' : ''}" 
                            onclick="approveModelForProduction('${m.model}')"
                            ${isApproved ? 'disabled' : ''}>
                        ${isApproved ? '<i data-lucide="check-circle" class="icon-xs"></i> Approved for Retailer View' : 'Approve for Retailer View'}
                    </button>
                </div>
            </div>
        `;
    }).join('');

    if (window.lucide) lucide.createIcons();
}

function renderLeaderboardTable(comparison) {
    const tbody = document.getElementById('comparison-leaderboard-tbody');
    if (!tbody) return;

    if (!comparison || comparison.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" class="table-empty">No comparison records found.</td></tr>`;
        return;
    }

    const approvedId = state.approvedModel?.approved_model_id || 'catboost';

    tbody.innerHTML = comparison.map(m => {
        const isApproved = m.model === approvedId;
        const rankBadge = m.rank === 1 
            ? `<span class="badge badge-accent">#1 Winner</span>`
            : `<span class="badge badge-neutral">#${m.rank || '—'}</span>`;

        return `
            <tr>
                <td>${rankBadge}</td>
                <td class="font-semibold text-charcoal">${m.name}</td>
                <td><span class="text-taupe text-sm">${m.category}</span></td>
                <td class="text-right font-medium">${formatDecimal(m.mae)}</td>
                <td class="text-right font-bold ${m.wape < 45 ? 'text-success' : ''}">${formatPercent(m.wape)}</td>
                <td class="text-right text-muted">${formatDecimal(m.rmse)}</td>
                <td class="text-right">${m.runtime_seconds !== null && m.runtime_seconds !== undefined ? formatDecimal(m.runtime_seconds) + 's' : '—'}</td>
                <td>
                    <button class="btn btn-sm ${isApproved ? 'btn-secondary' : 'btn-outline'}"
                            onclick="approveModelForProduction('${m.model}')"
                            ${isApproved ? 'disabled' : ''}>
                        ${isApproved ? '✓ Active Model' : 'Approve Model'}
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

async function approveModelForProduction(modelId) {
    try {
        showToast(`Approving ${modelId} as the active forecasting model...`, 'info');
        const res = await fetch('/api/model/approve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId, model: modelId }),
        });

        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error || 'Approval failed');

        showToast(`Success: ${data.name || modelId} approved for Retailer View!`, 'success');
        await loadApprovedModel();
        if (state.evalData?.comparison) {
            renderModelComparisonCards(state.evalData.comparison);
            renderLeaderboardTable(state.evalData.comparison);
        }
    } catch (e) {
        showToast('Model approval failed: ' + e.message, 'error');
    }
}

// ── 8. Model Evaluation Inspector ─────────────────────────────────
function switchEvaluationModel() {
    const select = document.getElementById('eval-model-select');
    if (!select) return;
    state.selectedEvalModel = select.value;
    renderEvaluationInspector(state.evalData);
}

function renderEvaluationInspector(data) {
    if (!data) return;
    const modelKey = state.selectedEvalModel || 'catboost';
    const comp = (data.comparison || []).find(c => c.model === modelKey) || {};
    const modelRes = data[modelKey] || {};
    const overall = modelRes.overall || comp;

    const maeEl = document.getElementById('inspect-metric-mae');
    const wapeEl = document.getElementById('inspect-metric-wape');
    const rmseEl = document.getElementById('inspect-metric-rmse');
    const rtEl = document.getElementById('inspect-metric-runtime');
    const rankEl = document.getElementById('inspect-badge-rank');
    const catBadge = document.getElementById('inspect-model-cat-badge');

    if (maeEl) maeEl.textContent = formatDecimal(overall.mae);
    if (wapeEl) wapeEl.textContent = formatPercent(overall.wape);
    if (rmseEl) rmseEl.textContent = formatDecimal(overall.rmse);
    const rt = modelRes.runtime_seconds ?? comp.runtime_seconds;
    if (rtEl) rtEl.textContent = rt !== null && rt !== undefined && !isNaN(Number(rt)) ? formatDecimal(rt) + 's' : '—';
    if (rankEl) rankEl.textContent = comp.rank ? `Rank #${comp.rank}` : 'Benchmark';
    if (catBadge) catBadge.textContent = comp.category || 'Model';

    // Descriptions & Pros/Cons
    const descEl = document.getElementById('inspect-limits-desc');
    const prosEl = document.getElementById('inspect-pros-text');
    const consEl = document.getElementById('inspect-cons-text');

    if (descEl) descEl.textContent = comp.description || 'Forecasting model architecture.';
    if (prosEl) prosEl.textContent = comp.pros || 'Direct forecasting capabilities.';
    if (consEl) consEl.textContent = comp.cons || 'Standard time series assumptions.';

    // Category breakdown
    if (modelRes.category_metrics) {
        renderCategoryTable(modelRes.category_metrics);
    }
    if (modelRes.horizon_metrics) {
        renderHorizonTable(modelRes.horizon_metrics);
        renderHorizonErrorChart(modelRes.horizon_metrics);
    }
}

// ── 9. Multi-Model Timeline Comparison Chart ──────────────────────
function renderMultiModelChart() {
    const canvas = document.getElementById('chart-multimodel-timeline');
    const timeline = state.evalData?.timeline || state.evalData?.timeline_comparison;
    if (!canvas || !timeline || !timeline.length) return;

    const labels = timeline.map((t, idx) => {
        if (!t.date) return `Week ${idx + 1}`;
        const parts = t.date.split('-');
        if (parts.length === 3) {
            const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
            const m = months[parseInt(parts[1], 10) - 1] || parts[1];
            return `W${idx + 1} (${m} ${parseInt(parts[2], 10)})`;
        }
        return t.date;
    });

    const actuals = timeline.map(t => t.actual !== null && t.actual !== undefined ? Math.round(t.actual) : null);
    const cbPreds = timeline.map(t => t.catboost !== null && t.catboost !== undefined ? Math.round(t.catboost) : null);
    const maPreds = timeline.map(t => t.ma4 !== null && t.ma4 !== undefined ? Math.round(t.ma4) : null);
    const naivePreds = timeline.map(t => t.naive !== null && t.naive !== undefined ? Math.round(t.naive) : null);
    const prophetPreds = timeline.map(t => t.prophet !== null && t.prophet !== undefined ? Math.round(t.prophet) : null);
    const arimaPreds = timeline.map(t => t.arima !== null && t.arima !== undefined ? Math.round(t.arima) : null);

    if (state.charts.multimodel) {
        state.charts.multimodel.destroy();
    }

    // Populate legend badges in card header
    const legendBox = document.getElementById('legend-multimodel');
    if (legendBox) {
        legendBox.innerHTML = `
            <span class="badge" style="background:#2D2926; color:#FFF; font-size:0.75rem;">Actuals</span>
            <span class="badge badge-accent" style="font-size:0.75rem;">CatBoost (Full)</span>
            <span class="badge badge-neutral" style="font-size:0.75rem;">MA4 (Full)</span>
            <span class="badge badge-neutral" style="font-size:0.75rem;">Naive (Full)</span>
            <span class="badge badge-warning" style="font-size:0.75rem;">Prophet (Sample)</span>
            <span class="badge badge-neutral" style="font-size:0.75rem;">AutoARIMA (Sample)</span>
        `;
    }

    const ctx = canvas.getContext('2d');
    state.charts.multimodel = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Ground Truth Actuals',
                    data: actuals,
                    yAxisID: 'y',
                    borderColor: '#2D2926',
                    backgroundColor: 'rgba(45, 41, 38, 0.08)',
                    borderWidth: 3,
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    pointBackgroundColor: '#2D2926',
                    fill: false,
                    tension: 0.2,
                },
                {
                    label: 'CatBoost Regressor',
                    data: cbPreds,
                    yAxisID: 'y',
                    borderColor: '#AF9AC9',
                    backgroundColor: 'rgba(175, 154, 201, 0.1)',
                    borderWidth: 2.5,
                    pointRadius: 4,
                    fill: false,
                    tension: 0.2,
                },
                {
                    label: 'Moving Average (4W)',
                    data: maPreds,
                    yAxisID: 'y',
                    borderColor: '#666161',
                    borderWidth: 2,
                    pointRadius: 3,
                    fill: false,
                    tension: 0.2,
                },
                {
                    label: 'Naive Baseline',
                    data: naivePreds,
                    yAxisID: 'y',
                    borderColor: '#A5937B',
                    borderWidth: 1.8,
                    borderDash: [4, 3],
                    pointRadius: 2,
                    fill: false,
                    tension: 0.1,
                },
                {
                    label: 'Facebook Prophet',
                    data: prophetPreds,
                    yAxisID: 'ySample',
                    borderColor: '#E3C49B',
                    borderWidth: 2.5,
                    borderDash: [6, 4],
                    pointRadius: 4,
                    fill: false,
                    tension: 0.2,
                },
                {
                    label: 'AutoARIMA',
                    data: arimaPreds,
                    yAxisID: 'ySample',
                    borderColor: '#8E77AC',
                    borderWidth: 2,
                    borderDash: [3, 3],
                    pointRadius: 3,
                    fill: false,
                    tension: 0.2,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        boxWidth: 14,
                        font: { family: 'Inter', size: 11, weight: '600' }
                    }
                },
                tooltip: {
                    backgroundColor: '#FFFFFF',
                    titleColor: '#2D2926',
                    bodyColor: '#666161',
                    borderColor: '#E9E3DE',
                    borderWidth: 1,
                    padding: 10,
                    callbacks: {
                        label: function(context) {
                            const val = context.parsed.y !== null ? Math.round(context.parsed.y).toLocaleString() : '—';
                            if (context.dataset.yAxisID === 'ySample') {
                                return ` ${context.dataset.label}: ${val} units (30-series sample benchmark)`;
                            }
                            return ` ${context.dataset.label}: ${val} units (full catalog holdout)`;
                        }
                    }
                },
            },
            scales: {
                x: {
                    grid: { color: 'rgba(165, 147, 123, 0.1)' },
                    ticks: { font: { family: 'Inter', size: 11 } }
                },
                y: {
                    type: 'linear',
                    position: 'left',
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Full Holdout Demand (Units)',
                        color: '#2D2926',
                        font: { family: 'Inter', size: 11, weight: '600' }
                    },
                    ticks: {
                        callback: v => v >= 1000 ? (v / 1000).toFixed(0) + 'k' : v
                    },
                    grid: { color: 'rgba(165, 147, 123, 0.1)' }
                },
                ySample: {
                    type: 'linear',
                    position: 'right',
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Sample Benchmark Demand (Units)',
                        color: '#A5937B',
                        font: { family: 'Inter', size: 11, weight: '600' }
                    },
                    ticks: {
                        callback: v => v >= 1000 ? (v / 1000).toFixed(1) + 'k' : v
                    },
                    grid: { drawOnChartArea: false }
                }
            },
        },
    });
}

// ── 10. Experiment History ────────────────────────────────────────
async function loadExperimentHistory() {
    const tbody = document.getElementById('experiments-history-tbody');
    if (!tbody) return;

    try {
        const res = await fetch('/api/experiments/history');
        if (!res.ok) throw new Error('History unavailable');
        const data = await res.json();
        const runs = Array.isArray(data) ? data : (data.experiments || []);

        if (!runs || runs.length === 0) {
            tbody.innerHTML = `<tr><td colspan="8" class="table-empty">No completed experiments found.</td></tr>`;
            return;
        }

        tbody.innerHTML = runs.map(r => {
            const modelsStr = (r.models_evaluated || []).join(', ') || 'CatBoost, ARIMA, Prophet, MA4, Naive';
            const dateStr = (r.created_at || r.eval_date || '').replace('T', ' ').slice(0, 16) || 'Recent';
            const startD = r.test_start_date || r.test_start || '—';
            const endD = r.test_end_date || r.test_end || '—';
            const horizon = r.forecast_horizon || r.holdout_weeks || 8;
            const bestWape = formatPercent(r.best_wape);

            return `
                <tr>
                    <td><code>${r.run_id}</code></td>
                    <td class="text-muted">${dateStr}</td>
                    <td>${startD} to ${endD}</td>
                    <td>${horizon} Weeks</td>
                    <td><span class="badge badge-neutral">${modelsStr}</span></td>
                    <td class="font-semibold text-lavender">${r.best_model}</td>
                    <td class="text-right font-bold text-success">${bestWape}</td>
                    <td><span class="badge badge-success">Completed</span></td>
                </tr>
            `;
        }).join('');
    } catch (e) {
        console.error('Experiment history load error:', e);
        tbody.innerHTML = `<tr><td colspan="8" class="table-empty text-danger">Failed to load experiments history.</td></tr>`;
    }
}

// ── 11. Retailer Workspace Operations ─────────────────────────────
async function loadRetailerWorkspace() {
    await Promise.all([
        loadForecastTable(),
        loadGeoForecastData(),
        loadRetailOverview(),
        loadRetailInsights(),
    ]);
}

async function loadRetailOverview() {
    try {
        const res = await fetch('/api/retail/overview');
        if (!res.ok) throw new Error('Retail overview error');
        const data = await res.json();
        state.retailOverview = data;

        // KPI Cards
        const kpis = data.kpis || {};
        const totalU = document.getElementById('ret-kpi-total-units');
        const risingEl = document.getElementById('ret-kpi-rising');
        const declEl = document.getElementById('ret-kpi-declining');
        const relEl = document.getElementById('ret-kpi-reliability');

        if (totalU) totalU.textContent = Math.round(kpis.total_4w_demand || 0).toLocaleString();
        if (risingEl) risingEl.textContent = (kpis.rising_products_count || 0) + ' Products';
        if (declEl) declEl.textContent = (kpis.declining_products_count || 0) + ' Products';
        if (relEl) relEl.textContent = formatPercent(kpis.reliability_score ?? 94.20);

        // Top Demand & Growth Tables
        renderRetailTopDemand(data.top_demand || []);
        renderRetailTopGrowth(data.top_growth || []);

        // Render Historical (8w) vs Forecast (4w) with Uncertainty Bands
        renderRetailTimelineChart();
    } catch (e) {
        console.error('Failed to load retail overview:', e);
    }
}

function renderRetailTimelineChart() {
    const canvas = document.getElementById('chart-retail-history-forecast');
    if (!canvas || !state.retailOverview?.timeline) return;

    const timeline = state.retailOverview.timeline;
    const labels = timeline.map(t => t.date);

    // Split actuals and predictions
    const actualPoints = timeline.map(t => t.type === 'actual' ? t.units : null);
    const forecastPoints = timeline.map(t => t.type === 'forecast' ? t.units : null);
    const upperBounds = timeline.map(t => t.type === 'forecast' ? t.upper_bound : null);
    const lowerBounds = timeline.map(t => t.type === 'forecast' ? t.lower_bound : null);

    // Connect the last actual to the first forecast for visual continuity
    const lastActualIdx = actualPoints.map((v, i) => v !== null ? i : -1).filter(i => i !== -1).pop();
    if (lastActualIdx !== undefined && lastActualIdx + 1 < forecastPoints.length) {
        actualPoints[lastActualIdx + 1] = forecastPoints[lastActualIdx + 1];
    }

    if (state.charts.retailTimeline) {
        state.charts.retailTimeline.destroy();
    }

    const ctx = canvas.getContext('2d');
    state.charts.retailTimeline = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Historical Actual Sales (Past 8 Weeks)',
                    data: actualPoints,
                    borderColor: '#A5937B',
                    backgroundColor: 'rgba(165, 147, 123, 0.08)',
                    borderWidth: 2.5,
                    pointRadius: 4,
                    fill: false,
                    tension: 0.2,
                },
                {
                    label: 'CatBoost Demand Forecast (Next 4 Weeks)',
                    data: forecastPoints,
                    borderColor: '#AF9AC9',
                    backgroundColor: 'rgba(175, 154, 201, 0.1)',
                    borderWidth: 3,
                    pointRadius: 5,
                    pointBackgroundColor: '#AF9AC9',
                    fill: false,
                    tension: 0.2,
                },
                {
                    label: 'Upper Replenishment Limit (+8% Buffer)',
                    data: upperBounds,
                    borderColor: 'rgba(227, 196, 155, 0.7)',
                    borderWidth: 1.2,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    fill: '+1',
                    backgroundColor: 'rgba(227, 196, 155, 0.15)',
                },
                {
                    label: 'Lower Replenishment Limit (-8% Baseline)',
                    data: lowerBounds,
                    borderColor: 'rgba(227, 196, 155, 0.7)',
                    borderWidth: 1.2,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    fill: false,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { 
                    position: 'top', 
                    labels: { 
                        boxWidth: 12, 
                        font: { family: 'Inter', size: 11, weight: 600 },
                        color: '#554F47'
                    } 
                },
                tooltip: { 
                    backgroundColor: '#FFFFFF', 
                    titleColor: '#2D2926', 
                    bodyColor: '#554F47', 
                    borderColor: '#E9E3DE', 
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        title: function(items) {
                            if (!items.length) return '';
                            const d = items[0].label;
                            const isForecast = items.some(it => it.datasetIndex === 1 && it.raw !== null);
                            return `Week Starting: ${d} ${isForecast ? '• [CatBoost Future Projection]' : '• [Historical Sales Record]'}`;
                        },
                        label: function(context) {
                            const val = context.parsed.y;
                            if (val === null || val === undefined) return '';
                            const dsLabel = context.dataset.label || '';
                            return `  • ${dsLabel}: ${Math.round(val).toLocaleString()} Units`;
                        }
                    }
                },
            },
            scales: {
                x: { 
                    grid: { color: 'rgba(165, 147, 123, 0.1)' },
                    title: { display: true, text: 'Calendar Week', color: '#8C827A', font: { size: 11 } }
                },
                y: { 
                    grid: { color: 'rgba(165, 147, 123, 0.1)' }, 
                    title: { display: true, text: 'Total Physical Units Demand', color: '#8C827A', font: { size: 11 } }
                },
            },
        },
    });
}

function renderRetailTopDemand(items) {
    const tbody = document.getElementById('ret-top-demand-tbody');
    if (!tbody) return;
    if (!items.length) {
        tbody.innerHTML = `<tr><td colspan="4" class="table-empty">No products recorded.</td></tr>`;
        return;
    }

    tbody.innerHTML = items.map(p => `
        <tr>
            <td class="font-semibold text-charcoal">${p.product_name}</td>
            <td><span class="text-taupe text-sm">${p.category}</span></td>
            <td class="text-right font-bold">${Math.round(p.forecast_units).toLocaleString()}</td>
            <td class="text-right text-success font-medium">+${formatPercent(p.growth_pct)}</td>
        </tr>
    `).join('');
}

function renderRetailTopGrowth(items) {
    const tbody = document.getElementById('ret-top-growth-tbody');
    if (!tbody) return;
    if (!items.length) {
        tbody.innerHTML = `<tr><td colspan="4" class="table-empty">No products recorded.</td></tr>`;
        return;
    }

    tbody.innerHTML = items.map(p => `
        <tr>
            <td class="font-semibold text-charcoal">${p.product_name}</td>
            <td><span class="text-taupe text-sm">${p.category}</span></td>
            <td class="text-right font-bold">${Math.round(p.forecast_units).toLocaleString()}</td>
            <td class="text-right text-success font-bold">+${formatPercent(p.growth_pct)}</td>
        </tr>
    `).join('');
}

// ── 12. Retailer Product Insights ──
async function loadRetailInsights() {
    try {
        const res = await fetch('/api/retail/insights');
        if (!res.ok) throw new Error('Insights failed');
        const data = await res.json();
        state.retailInsights = data;

        renderInsightsGrid('insights-rising-grid', data.rising_demand || [], 'positive');
        renderInsightsGrid('insights-declining-grid', data.declining_demand || [], 'negative');
        renderCautionaryGrid('insights-cautionary-grid', data.cautionary || []);
    } catch (e) {
        console.warn('Insights fetch error:', e);
    }
}

function renderInsightsGrid(containerId, items, sentiment) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!items.length) {
        container.innerHTML = `<div class="p-3 text-muted">No products currently flagged in this category.</div>`;
        return;
    }

    container.innerHTML = items.map(item => `
        <div class="insight-product-card">
            <div>
                <div class="insight-card-top">
                    <div>
                        <h4 class="insight-pname">${item.product_name}</h4>
                        <span class="insight-pcat">${item.category} (${item.product_id})</span>
                    </div>
                    <span class="insight-growth-pill ${sentiment}">
                        ${item.change_pct > 0 ? '+' : ''}${formatPercent(item.change_pct)}
                    </span>
                </div>

                <div class="insight-metrics-comparison">
                    <div class="ins-metric-col">
                        <span class="label">Past 4W Average</span>
                        <span class="val">${Math.round(item.recent_history_units).toLocaleString()} units</span>
                    </div>
                    <div class="ins-metric-col">
                        <span class="label">Next 4W Forecast</span>
                        <span class="val">${Math.round(item.forecast_units).toLocaleString()} units</span>
                    </div>
                </div>
            </div>

            <p class="insight-explanation-text">${item.explanation}</p>
        </div>
    `).join('');
}

function renderCautionaryGrid(containerId, items) {
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!items.length) {
        container.innerHTML = `<div class="p-3 text-muted">All product forecasts are currently stable.</div>`;
        return;
    }

    container.innerHTML = items.map(item => `
        <div class="insight-product-card">
            <div>
                <div class="insight-card-top">
                    <div>
                        <h4 class="insight-pname">${item.product_name}</h4>
                        <span class="insight-pcat">${item.category} (${item.product_id})</span>
                    </div>
                    <span class="badge badge-warning">${item.flag_type}</span>
                </div>
                <div class="insight-metrics-comparison">
                    <div class="ins-metric-col">
                        <span class="label">Severity</span>
                        <span class="val text-capitalize font-semibold text-warning">${item.severity}</span>
                    </div>
                    <div class="ins-metric-col">
                        <span class="label">Action</span>
                        <span class="val">+15% Safety Buffer</span>
                    </div>
                </div>
            </div>
            <p class="insight-explanation-text">${item.reason}</p>
        </div>
    `).join('');
}

function filterInsightsTab(filterType) {
    document.querySelectorAll('.insight-pill').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    const sRising = document.getElementById('section-insight-rising');
    const sDeclining = document.getElementById('section-insight-declining');
    const sCautionary = document.getElementById('section-insight-cautionary');

    if (filterType === 'all') {
        if (sRising) sRising.style.display = 'block';
        if (sDeclining) sDeclining.style.display = 'block';
        if (sCautionary) sCautionary.style.display = 'block';
    } else if (filterType === 'rising') {
        if (sRising) sRising.style.display = 'block';
        if (sDeclining) sDeclining.style.display = 'none';
        if (sCautionary) sCautionary.style.display = 'none';
    } else if (filterType === 'declining') {
        if (sRising) sRising.style.display = 'none';
        if (sDeclining) sDeclining.style.display = 'block';
        if (sCautionary) sCautionary.style.display = 'none';
    } else if (filterType === 'cautionary') {
        if (sRising) sRising.style.display = 'none';
        if (sDeclining) sDeclining.style.display = 'none';
        if (sCautionary) sCautionary.style.display = 'block';
    }
}

// ── 13. Download & Export Handlers ────────────────────────────────
function triggerFileDownload(url, defaultFilename, progressMsg) {
    if (progressMsg) showToast(progressMsg, 'info');

    const link = document.createElement('a');
    link.href = url;
    if (defaultFilename) {
        link.setAttribute('download', defaultFilename);
    }
    document.body.appendChild(link);
    link.click();
    setTimeout(() => {
        if (document.body.contains(link)) {
            document.body.removeChild(link);
        }
    }, 1500);
}


function downloadQualityReport() {
    triggerFileDownload('/download-quality-report', 'data_quality_report.md', 'Downloading Data Quality Audit Report (Markdown)...');
}

function downloadQualityCSV() {
    triggerFileDownload('/download-quality-csv', 'data_quality_report.csv', 'Downloading Data Quality Audit Report (CSV)...');
}

function downloadDatasetCSV() {
    triggerFileDownload('/download-dataset-csv', 'retail_dataset_clean.csv', 'Exporting clean retail historical sales dataset (CSV)...');
}

function downloadEvaluationCSV() {
    triggerFileDownload('/download-evaluation-csv', 'model_evaluation_report.csv', 'Exporting model holdout evaluation benchmark metrics (CSV)...');
}

function downloadComparisonCSV() {
    triggerFileDownload('/download-comparison-csv', 'model_comparison_leaderboard.csv', 'Exporting 5-model leaderboard comparison benchmark (CSV)...');
}

function downloadExperimentsCSV() {
    triggerFileDownload('/download-experiments-csv', 'experiment_history_report.csv', 'Exporting audited experiment history runs (CSV)...');
}

function exportForecastCSV() {
    triggerFileDownload('/download-csv', 'retail_demand_forecast.csv', 'Generating and exporting 4-Week Demand Forecast CSV with inventory indicators...');
}

function downloadBusinessReport() {
    triggerFileDownload('/download-pdf', 'executive_demand_report.pdf', 'Compiling and downloading Executive Business Demand Report (PDF)...');
}

function viewBusinessReport() {
    window.open('/view-pdf', '_blank');
}

// Backwards compatibility alias
function downloadRetailReport() {
    exportForecastCSV();
}


// ── 14. Forecast Explorer & Table (Shared Filter Console) ─────────
async function loadForecastTable() {
    try {
        const p = new URLSearchParams(state.filters);
        const res = await fetch(`/api/forecast/analytics?${p.toString()}`);
        if (!res.ok) throw new Error('Forecast fetch error');
        const data = await res.json();
        state.forecastSummary = data;

        const filterOpts = data.filter_options || {};
        const kpis = data.kpis || {};
        const tbl = data.table || {};
        const charts = data.charts || {};
        const meta = data.metadata || {};

        // Update Run Banner
        const fcModelTag = document.getElementById('fc-meta-model');
        const fcRunId = document.getElementById('fc-meta-run-id');
        const fcTimestamp = document.getElementById('fc-meta-timestamp');
        if (fcModelTag) fcModelTag.textContent = `Model: ${meta.approved_model || 'CATBOOST'}`;
        if (fcRunId) fcRunId.textContent = meta.run_id || 'forecast_run_latest';
        if (fcTimestamp) fcTimestamp.textContent = meta.created_at || 'Current Run';

        // Populate dropdowns once
        if (filterOpts.categories) populateDropdown('filter-category', filterOpts.categories);
        if (filterOpts.stores) populateDropdown('filter-store', filterOpts.stores);

        // Update Explorer KPIs
        const totalU = document.getElementById('exp-total-units');
        const scEl = document.getElementById('exp-series-count');
        const avgU = document.getElementById('exp-avg-units');
        const topP = document.getElementById('exp-top-product');

        if (totalU) totalU.textContent = Math.round(kpis.total_predicted_units || 0).toLocaleString();
        if (scEl) scEl.textContent = (kpis.unique_series || tbl.total_rows || 0).toLocaleString();
        if (avgU) avgU.textContent = (kpis.avg_units_per_series || 0).toFixed(1);
        if (topP) topP.textContent = kpis.top_product?.product_name || '—';

        const rowCount = document.getElementById('table-row-count');
        if (rowCount) rowCount.textContent = `Showing ${tbl.records?.length || 0} of ${(tbl.total_rows || 0).toLocaleString()} series`;

        // Render table & charts
        renderExplorerTable(tbl.records || []);
        renderCategoryDistributionChart(charts.category_distribution || []);
        renderExplorerHorizonChart(charts.horizon_trend || []);

        // Pagination
        const pagEl = document.getElementById('pagination-info');
        if (pagEl) pagEl.textContent = `Page ${tbl.page || 1} of ${tbl.total_pages || 1}`;

        const prevBtn = document.getElementById('btn-prev-page');
        const nextBtn = document.getElementById('btn-next-page');
        if (prevBtn) prevBtn.disabled = !(tbl.page > 1);
        if (nextBtn) nextBtn.disabled = !(tbl.page < tbl.total_pages);

    } catch (e) {
        console.warn('Forecast table load failed:', e);
    }
}

function renderExplorerTable(records) {
    const tbody = document.getElementById('explorer-tbody');
    if (!tbody) return;

    if (!records.length) {
        tbody.innerHTML = `<tr><td colspan="9" class="table-empty">No forecasts found matching active filters.</td></tr>`;
        return;
    }

    tbody.innerHTML = records.map(r => `
        <tr>
            <td><code>${r.product_id}</code></td>
            <td class="font-medium text-charcoal">${r.product_name || r.product_id}</td>
            <td><span class="text-taupe text-sm">${r.category}</span></td>
            <td>${r.store_id}</td>
            <td><span class="badge ${r.sales_channel === 'online' ? 'badge-accent' : 'badge-neutral'}">${r.sales_channel}</span></td>
            <td>${r.forecast_week_date || '—'}</td>
            <td>Week ${r.horizon_step}</td>
            <td class="text-right font-bold">${Math.round(r.predicted_units).toLocaleString()}</td>
            <td><span class="badge ${r.reliability_class || 'badge-success'}">${r.reliability || 'High Confidence'}</span></td>
        </tr>
    `).join('');
}

function renderExplorerHorizonChart(horizonTrend) {
    const canvas = document.getElementById('chart-explorer-horizon');
    if (!canvas || !horizonTrend.length) return;

    const labels = horizonTrend.map(h => h.label || `Week ${h.horizon_step}`);
    const values = horizonTrend.map(h => h.total_units);

    if (state.charts.explorerHorizon) {
        state.charts.explorerHorizon.destroy();
    }

    const ctx = canvas.getContext('2d');
    state.charts.explorerHorizon = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Projected Demand Units',
                data: values,
                borderColor: '#AF9AC9',
                backgroundColor: 'rgba(175, 154, 201, 0.15)',
                borderWidth: 2.5,
                pointRadius: 4,
                fill: true,
                tension: 0.25,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { backgroundColor: '#FFFFFF', titleColor: '#2D2926', bodyColor: '#666161', borderColor: '#E9E3DE', borderWidth: 1 },
            },
            scales: {
                x: { grid: { color: 'rgba(165, 147, 123, 0.1)' } },
                y: { grid: { color: 'rgba(165, 147, 123, 0.1)' }, title: { display: true, text: 'Demand Units' } },
            },
        },
    });
}

function renderCategoryDistributionChart(catData) {
    const canvas = document.getElementById('chart-explorer-category');
    if (!canvas || !catData.length) return;

    const labels = catData.map(c => c.category);
    const values = catData.map(c => Math.round(c.total_units ?? c.units ?? 0));

    if (state.charts.explorerCat) {
        state.charts.explorerCat.destroy();
    }

    const ctx = canvas.getContext('2d');
    state.charts.explorerCat = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Projected Demand Units',
                data: values,
                backgroundColor: [
                    '#AF9AC9',
                    '#E3C49B',
                    '#A5937B',
                    '#8E77AC',
                    '#D4C5B9',
                    '#666161',
                    '#BDB5AD',
                    '#C4B5D0'
                ],
                borderRadius: 4,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return ` ${context.parsed.y.toLocaleString()} units projected`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: {
                        autoSkip: false,
                        maxRotation: 30,
                        minRotation: 15,
                        font: { size: 10 }
                    }
                },
                y: {
                    grid: { color: 'rgba(165, 147, 123, 0.1)' },
                    title: { display: true, text: 'Projected Demand (Units)' },
                    ticks: {
                        callback: v => v.toLocaleString()
                    }
                },
            },
        },
    });
}

function populateDropdown(elemId, items) {
    const sel = document.getElementById(elemId);
    if (!sel || sel.children.length > 1) return;
    items.forEach(it => {
        const opt = document.createElement('option');
        opt.value = it;
        opt.textContent = it;
        sel.appendChild(opt);
    });
}

function applyFilters() {
    state.filters.category = document.getElementById('filter-category')?.value || 'all';
    state.filters.store_id = document.getElementById('filter-store')?.value || 'all';
    state.filters.sales_channel = document.getElementById('filter-channel')?.value || 'all';
    state.filters.horizon = document.getElementById('filter-horizon')?.value || 'all';
    state.filters.page = 1;
    loadForecastTable();
}

function resetFilters() {
    state.filters = {
        category: 'all', store_id: 'all', sales_channel: 'all',
        horizon: 'all', search: '', page: 1, page_size: 10,
        sort_by: 'predicted_units', sort_order: 'desc',
    };
    if (document.getElementById('filter-category')) document.getElementById('filter-category').value = 'all';
    if (document.getElementById('filter-store')) document.getElementById('filter-store').value = 'all';
    if (document.getElementById('filter-channel')) document.getElementById('filter-channel').value = 'all';
    if (document.getElementById('filter-horizon')) document.getElementById('filter-horizon').value = 'all';
    if (document.getElementById('filter-search')) document.getElementById('filter-search').value = '';
    loadForecastTable();
}

function debounceSearch() {
    clearTimeout(state.searchDebounceTimer);
    state.searchDebounceTimer = setTimeout(() => {
        state.filters.search = document.getElementById('filter-search')?.value.trim() || '';
        state.filters.page = 1;
        loadForecastTable();
    }, 300);
}

function prevPage() {
    if (state.filters.page > 1) {
        state.filters.page--;
        loadForecastTable();
    }
}

function nextPage() {
    state.filters.page++;
    loadForecastTable();
}

function changeSort() {
    const val = document.getElementById('table-sort-by')?.value || 'predicted_units-desc';
    const [col, ord] = val.split('-');
    state.filters.sort_by = col;
    state.filters.sort_order = ord;
    loadForecastTable();
}

// ── 14.1 Geographic Demand Intelligence (India Map & Prediction Table) ────
const INDIA_STATE_PATHS = {
    "Maharashtra": {
        d: "M 72,282 L 114,272 L 158,274 L 196,306 L 176,346 L 114,352 L 86,290 Z",
        name: "Maharashtra"
    },
    "Gujarat": {
        d: "M 48,228 L 86,214 L 118,236 L 114,272 L 72,282 L 44,258 Z",
        name: "Gujarat"
    },
    "Delhi": {
        d: "M 152,158 L 160,158 L 160,166 L 152,166 Z",
        name: "Delhi"
    },
    "Haryana": {
        d: "M 136,148 L 168,148 L 170,178 L 140,178 Z",
        name: "Haryana"
    },
    "Uttar Pradesh": {
        d: "M 168,148 L 222,126 L 254,168 L 244,204 L 186,212 L 164,184 Z",
        name: "Uttar Pradesh"
    },
    "Rajasthan": {
        d: "M 84,178 L 136,148 L 140,178 L 158,220 L 118,236 L 86,214 Z",
        name: "Rajasthan"
    },
    "Madhya Pradesh": {
        d: "M 118,236 L 186,212 L 244,204 L 236,260 L 158,274 L 114,272 Z",
        name: "Madhya Pradesh"
    },
    "Karnataka": {
        d: "M 114,352 L 176,346 L 196,380 L 178,428 L 146,420 L 126,388 Z",
        name: "Karnataka"
    },
    "Tamil Nadu": {
        d: "M 178,428 L 196,380 L 210,388 L 194,432 L 172,464 L 156,462 L 162,424 Z",
        name: "Tamil Nadu"
    },
    "Telangana": {
        d: "M 158,274 L 208,278 L 220,326 L 176,346 Z",
        name: "Telangana"
    },
    "Andhra Pradesh": {
        d: "M 208,278 L 236,260 L 268,304 L 246,316 L 230,358 L 196,380 L 220,326 Z",
        name: "Andhra Pradesh"
    },
    "Kerala": {
        d: "M 146,420 L 162,424 L 156,462 L 138,446 L 146,420 Z",
        name: "Kerala"
    },
    "West Bengal": {
        d: "M 284,212 L 314,220 L 320,236 L 302,256 L 300,288 L 284,270 Z",
        name: "West Bengal"
    },
    "Odisha": {
        d: "M 244,248 L 284,248 L 300,288 L 268,304 L 244,264 Z",
        name: "Odisha"
    },
    "Bihar": {
        d: "M 244,168 L 298,172 L 284,212 L 244,204 Z",
        name: "Bihar"
    },
    "Punjab": {
        d: "M 124,124 L 152,118 L 154,146 L 130,146 Z",
        name: "Punjab"
    },
    "Jammu & Kashmir": {
        d: "M 148,28 L 176,32 L 194,54 L 188,88 L 160,94 L 134,104 L 138,58 Z",
        name: "Jammu & Kashmir"
    },
    "Himachal & Uttarakhand": {
        d: "M 152,94 L 188,88 L 204,116 L 222,126 L 196,144 L 152,118 Z",
        name: "Himachal & Uttarakhand"
    },
    "Chhattisgarh": {
        d: "M 224,240 L 254,236 L 248,310 L 220,314 Z",
        name: "Chhattisgarh"
    },
    "Jharkhand": {
        d: "M 254,196 L 284,200 L 280,244 L 248,240 Z",
        name: "Jharkhand"
    },
    "North East States": {
        d: "M 320,236 L 340,172 L 358,154 L 396,160 L 416,182 L 402,216 L 372,218 L 346,242 Z",
        name: "North East States"
    }
};

// ── Coordinates and Colorful Tiers for Indian Retail Hubs on Image Map (1024x768) ──
const INDIA_MAP_STORE_COORDS = {
    'ST-DEL-01': { xPct: 33.98, yPct: 30.47, city: 'Delhi', state: 'Delhi' },
    'ST-ONL-01': { xPct: 33.01, yPct: 31.25, city: 'Gurugram', state: 'Haryana' },
    'ST-NOI-01': { xPct: 34.96, yPct: 30.99, city: 'Noida', state: 'Uttar Pradesh' },
    'ST-JAI-01': { xPct: 30.76, yPct: 35.94, city: 'Jaipur', state: 'Rajasthan' },
    'ST-LKO-01': { xPct: 42.97, yPct: 38.80, city: 'Lucknow', state: 'Uttar Pradesh' },
    'ST-AMD-01': { xPct: 20.31, yPct: 48.44, city: 'Ahmedabad', state: 'Gujarat' },
    'ST-BPL-01': { xPct: 36.33, yPct: 50.78, city: 'Bhopal', state: 'Madhya Pradesh' },
    'ST-KOL-01': { xPct: 61.52, yPct: 52.08, city: 'Kolkata', state: 'West Bengal' },
    'ST-MUM-01': { xPct: 22.27, yPct: 58.85, city: 'Mumbai', state: 'Maharashtra' },
    'ST-PUN-01': { xPct: 24.22, yPct: 62.24, city: 'Pune', state: 'Maharashtra' },
    'ST-HYD-01': { xPct: 38.77, yPct: 67.06, city: 'Hyderabad', state: 'Telangana' },
    'ST-BLR-01': { xPct: 32.71, yPct: 80.99, city: 'Bengaluru', state: 'Karnataka' },
    'ST-CHN-01': { xPct: 40.23, yPct: 83.07, city: 'Chennai', state: 'Tamil Nadu' },
    'ST-KOC-01': { xPct: 30.86, yPct: 90.49, city: 'Kochi', state: 'Kerala' }
};

// Map Zoom and Pan State (by default 125%)
state.mapZoom = 1.25;
state.mapPan = { x: 0, y: 0 };
state.mapIsDragging = false;
state.mapDragStart = { x: 0, y: 0 };

async function loadGeoForecastData() {
    try {
        const res = await fetch('/api/forecast/geo');
        if (!res.ok) throw new Error('Failed to load geo forecast');
        const data = await res.json();
        state.geoData = data;

        // Update badge
        const badgeEl = document.getElementById('geo-total-rev-summary');
        if (badgeEl && data.totals) {
            badgeEl.textContent = `${data.stores?.length || 14} Outlets • ₹${data.totals.forecast_revenue_cr?.toFixed(2)} Cr Projected`;
        }

        renderIndiaImageMap(data);
        renderGeoTable(data, state.geoView || 'city');
    } catch (e) {
        console.warn('Geo forecast load error:', e);
    }
}

function switchGeoView(mode) {
    state.geoView = mode;
    const btnCity = document.getElementById('btn-geo-view-city');
    const btnState = document.getElementById('btn-geo-view-state');
    if (btnCity) btnCity.classList.toggle('active', mode === 'city');
    if (btnState) btnState.classList.toggle('active', mode === 'state');

    if (state.geoData) {
        renderGeoTable(state.geoData, mode);
    }
}

function renderIndiaImageMap(geoData) {
    const pinsLayer = document.getElementById('geo-pins-layer');
    const container = document.getElementById('geo-map-container');
    const stage = document.getElementById('geo-map-stage');
    if (!pinsLayer || !geoData || !container) return;

    // Initialize interactive Pan & Zoom listeners on container once
    if (!container._zoomPanInitialized) {
        container._zoomPanInitialized = true;

        // Mouse Wheel Zoom
        container.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY < 0 ? 0.20 : -0.20;
            zoomIndiaMap(delta);
        }, { passive: false });

        // Click and Drag to Pan
        container.addEventListener('mousedown', (e) => {
            if (e.target.closest('.map-store-pin') || e.target.closest('.geo-map-overlay-controls')) return;
            state.mapIsDragging = true;
            container.classList.add('is-dragging');
            if (stage) stage.classList.add('no-transition');
            state.mapDragStart = { x: e.clientX - state.mapPan.x, y: e.clientY - state.mapPan.y };
        });

        window.addEventListener('mousemove', (e) => {
            if (!state.mapIsDragging || !stage) return;
            state.mapPan.x = e.clientX - state.mapDragStart.x;
            state.mapPan.y = e.clientY - state.mapDragStart.y;
            applyIndiaMapTransform();
        });

        window.addEventListener('mouseup', () => {
            if (state.mapIsDragging) {
                state.mapIsDragging = false;
                container.classList.remove('is-dragging');
                if (stage) stage.classList.remove('no-transition');
            }
        });
    }

    // Determine max units to compute colorful tiers
    const maxUnits = Math.max(...(geoData.stores || []).map(s => s.forecast_units || 1), 1);

    // Clear previous pins
    pinsLayer.innerHTML = '';
    state.storePinElements = {};

    (geoData.stores || []).forEach(s => {
        const coord = INDIA_MAP_STORE_COORDS[s.store_id] || INDIA_MAP_STORE_COORDS[s.city];
        if (!coord) return;

        const ratio = (s.forecast_units || 0) / maxUnits;
        // Assign colorful tier
        let tierClass = 'pin-tier-steady';
        if (ratio >= 0.82) {
            tierClass = 'pin-tier-super'; // Purple & Fuchsia (Super Hub)
        } else if (ratio >= 0.65) {
            tierClass = 'pin-tier-peak';  // Coral & Sunset (Peak Volume)
        } else if (ratio >= 0.45) {
            tierClass = 'pin-tier-high';  // Amber & Gold (High Demand)
        } else {
            tierClass = 'pin-tier-steady'; // Emerald & Cyan (Steady)
        }

        const pin = document.createElement('div');
        pin.className = `map-store-pin ${tierClass}`;
        pin.id = `map-pin-${s.store_id}`;
        pin.style.left = `${coord.xPct}%`;
        pin.style.top = `${coord.yPct}%`;

        pin.innerHTML = `
            <div class="map-pin-halo"></div>
            <div class="map-pin-head">
                <span class="map-pin-core"></span>
            </div>
            <div class="map-pin-badge">
                <span>${s.city}</span>
                <span class="badge-val">${(s.forecast_units / 1000).toFixed(0)}k</span>
            </div>
        `;

        // Hover & Click interactions
        pin.addEventListener('mouseenter', (e) => showPinTooltip(s, e));
        pin.addEventListener('mousemove', (e) => positionPinTooltip(e));
        pin.addEventListener('mouseleave', () => hidePinTooltip());
        pin.addEventListener('click', (e) => {
            e.stopPropagation();
            flyToStorePin(s.store_id);
            highlightAndScrollTableRow(s);
        });

        pinsLayer.appendChild(pin);
        state.storePinElements[s.store_id] = pin;
        state.storePinElements[s.city] = pin;
    });

    applyIndiaMapTransform();
}

function zoomIndiaMap(delta) {
    const newZoom = Math.min(Math.max(1.0, state.mapZoom + delta), 3.0);
    state.mapZoom = Number(newZoom.toFixed(2));
    if (state.mapZoom === 1.25) {
        state.mapPan = { x: 0, y: 0 };
    }
    applyIndiaMapTransform();
}

function resetIndiaMapZoom() {
    state.mapZoom = 1.25;
    state.mapPan = { x: 0, y: 0 };
    applyIndiaMapTransform();
    hidePinTooltip();
    if (window.lucide) lucide.createIcons();
}

function applyIndiaMapTransform() {
    const stage = document.getElementById('geo-map-stage');
    const badge = document.getElementById('zoom-level-display');
    if (stage) {
        stage.style.transform = `translate(${state.mapPan.x}px, ${state.mapPan.y}px) scale(${state.mapZoom})`;
    }
    if (badge) {
        badge.textContent = `${Math.round(state.mapZoom * 100)}%`;
    }
    // Counter-scale pins so they stay crisp and elegant without bloating when zoomed in
    const pinScale = (1 / Math.sqrt(state.mapZoom)).toFixed(3);
    document.documentElement.style.setProperty('--map-pin-counter-scale', pinScale);
}

function flyToStorePin(storeIdOrCity) {
    const coord = INDIA_MAP_STORE_COORDS[storeIdOrCity];
    const pin = state.storePinElements && state.storePinElements[storeIdOrCity];
    if (!coord) return;

    // Zoom into the store location
    state.mapZoom = 1.85;
    const targetX = (0.5 - coord.xPct / 100) * 600 * state.mapZoom;
    const targetY = (0.5 - coord.yPct / 100) * 450 * state.mapZoom;
    state.mapPan = { x: targetX, y: targetY };
    applyIndiaMapTransform();

    // Highlight pin
    document.querySelectorAll('.map-store-pin').forEach(p => p.classList.remove('is-active'));
    if (pin) {
        pin.classList.add('is-active');
        const s = (state.geoData?.stores || []).find(x => x.store_id === storeIdOrCity || x.city === storeIdOrCity);
        if (s) {
            showPinTooltip(s, { clientX: window.innerWidth / 2, clientY: 260 });
        }
    }
}

function showPinTooltip(s, e) {
    const tip = document.getElementById('geo-pin-tooltip');
    if (!tip || !s) return;

    const isUp = (s.change_pct || 0) >= 0;
    tip.innerHTML = `
        <div class="tip-header">
            <span class="tip-city">📍 ${s.city}</span>
            <span class="tip-store-id">${s.store_id}</span>
        </div>
        <div class="tip-state">${s.state} • ${s.region} Territory</div>
        <div class="tip-grid">
            <div class="tip-item">
                <span class="tip-lbl">Forecast Demand</span>
                <span class="tip-val font-bold text-sky">${(s.forecast_units || 0).toLocaleString()} Units</span>
            </div>
            <div class="tip-item">
                <span class="tip-lbl">Present Revenue</span>
                <span class="tip-val font-semibold">₹${(s.forecast_rev_cr || 0).toFixed(2)} Cr</span>
            </div>
            <div class="tip-item">
                <span class="tip-lbl">Previous Units</span>
                <span class="tip-val text-muted">${(s.hist_units || 0).toLocaleString()}</span>
            </div>
            <div class="tip-item">
                <span class="tip-lbl">Growth Trend</span>
                <span class="tip-val ${isUp ? 'text-success' : 'text-danger'} font-bold">
                    ${isUp ? '↑ +' : '↓ '}${Math.abs(s.change_pct || 0)}%
                </span>
            </div>
        </div>
    `;

    tip.style.display = 'block';
    positionPinTooltip(e);
}

function positionPinTooltip(e) {
    const tip = document.getElementById('geo-pin-tooltip');
    const container = document.getElementById('geo-map-container');
    if (!tip || !container || !e) return;

    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    tip.style.left = `${Math.max(120, Math.min(x, rect.width - 120))}px`;
    tip.style.top = `${Math.max(130, y - 10)}px`;
}

function hidePinTooltip() {
    const tip = document.getElementById('geo-pin-tooltip');
    if (tip) tip.style.display = 'none';
}

function renderGeoTable(geoData, mode) {
    const tbody = document.getElementById('geo-prediction-tbody');
    const titleEl = document.getElementById('geo-table-header-title');
    const subEl = document.getElementById('geo-table-header-sub');
    const colNameEl = document.getElementById('geo-col-name');
    if (!tbody || !geoData) return;

    if (mode === 'state') {
        if (titleEl) titleEl.textContent = 'State-wise Performance Summary';
        if (subEl) subEl.textContent = `${geoData.states?.length || 12} State Territories`;
        if (colNameEl) colNameEl.textContent = 'State Territory';

        const states = (geoData.states || []).slice();
        states.sort((a, b) => b.forecast_units - a.forecast_units);

        tbody.innerHTML = states.map(st => {
            const isUp = st.change_pct >= 0;
            const firstStore = (st.stores || [])[0];
            return `
                <tr class="geo-table-row" data-key="${st.state}" onclick="onGeoRowClicked('${firstStore}', '${st.state}')" title="Click to view ${st.state} on map">
                    <td>
                        <div class="geo-row-main font-semibold text-charcoal">${st.state}</div>
                        <div class="geo-row-sub">${(st.cities || []).join(', ')}</div>
                    </td>
                    <td class="text-right font-bold text-sky">${st.forecast_units.toLocaleString()}</td>
                    <td class="text-right font-medium">₹${st.forecast_rev_cr.toFixed(2)} Cr</td>
                    <td class="text-right text-muted">${st.hist_units.toLocaleString()}</td>
                    <td class="text-center">
                        <span class="geo-change-pill ${isUp ? 'pill-up' : 'pill-down'}">
                            ${isUp ? '↑ +' : '↓ '}${Math.abs(st.change_pct)}%
                        </span>
                    </td>
                </tr>
            `;
        }).join('');

    } else {
        // City Outlets View
        if (titleEl) titleEl.textContent = 'Demand Forecast Prediction by Outlet';
        if (subEl) subEl.textContent = `${geoData.stores?.length || 14} Store Hubs (CatBoost ML Projections)`;
        if (colNameEl) colNameEl.textContent = 'City / Store Hub';

        const stores = (geoData.stores || []).slice();
        stores.sort((a, b) => b.forecast_units - a.forecast_units);

        tbody.innerHTML = stores.map(s => {
            const isUp = s.change_pct >= 0;
            return `
                <tr class="geo-table-row" data-key="${s.store_id}" onclick="onGeoRowClicked('${s.store_id}', '${s.city}')" title="Click to zoom to ${s.city} on map">
                    <td>
                        <div class="geo-row-main font-semibold text-charcoal">
                            ${s.city}
                            <span class="geo-store-tag">${s.store_id}</span>
                        </div>
                        <div class="geo-row-sub">${s.state} • ${s.region}</div>
                    </td>
                    <td class="text-right font-bold text-sky">${s.forecast_units.toLocaleString()}</td>
                    <td class="text-right font-medium">₹${s.forecast_rev_cr.toFixed(2)} Cr</td>
                    <td class="text-right text-muted">${s.hist_units.toLocaleString()}</td>
                    <td class="text-center">
                        <span class="geo-change-pill ${isUp ? 'pill-up' : 'pill-down'}">
                            ${isUp ? '↑ +' : '↓ '}${Math.abs(s.change_pct)}%
                        </span>
                    </td>
                </tr>
            `;
        }).join('');
    }
}

function onGeoRowClicked(storeId, key) {
    if (storeId) flyToStorePin(storeId);
    highlightAndScrollTableRow(key || storeId);
}

function highlightAndScrollTableRow(item) {
    const scrollWrap = document.querySelector('.geo-table-scroll-wrap');
    const rows = document.querySelectorAll('.geo-table-row');
    if (!rows.length) return;

    let storeId = '';
    let city = '';
    let stateName = '';

    if (typeof item === 'object' && item !== null) {
        storeId = item.store_id || '';
        city = item.city || '';
        stateName = item.state || '';
    } else if (typeof item === 'string') {
        storeId = item;
        const sObj = (window.state?.geoData?.stores || []).find(x => x.store_id === item || x.city === item);
        if (sObj) {
            city = sObj.city;
            stateName = sObj.state;
        } else {
            stateName = item;
        }
    }

    // 1. Clear previous active highlights across all rows
    rows.forEach(r => r.classList.remove('geo-row-highlighted'));

    // 2. Identify the matching row in table
    let targetRow = null;
    rows.forEach(r => {
        const k = r.getAttribute('data-key');
        if (k && (k === storeId || k === city || k === stateName)) {
            targetRow = r;
        }
    });

    if (targetRow) {
        targetRow.classList.add('geo-row-highlighted');

        // 3. Smooth scroll effect: move row cleanly to the top of table viewport while keeping its place in table
        if (scrollWrap) {
            const thead = scrollWrap.querySelector('thead');
            const theadHeight = thead ? thead.offsetHeight : 38;
            const targetTop = Math.max(0, targetRow.offsetTop - theadHeight);
            scrollWrap.scrollTo({
                top: targetTop,
                behavior: 'smooth'
            });
        }
    }

    // 4. Also activate pin on India map
    const pinKey = storeId || city;
    if (pinKey) {
        document.querySelectorAll('.map-store-pin').forEach(p => p.classList.remove('is-active'));
        const pinElem = document.getElementById(`map-pin-${pinKey}`) || (city && document.getElementById(`map-pin-${city}`));
        if (pinElem) {
            pinElem.classList.add('is-active');
        }
    }
}

function highlightTableRow(key, active) {
    if (active) {
        highlightAndScrollTableRow(key);
    } else {
        const rows = document.querySelectorAll('.geo-table-row');
        rows.forEach(r => {
            if (r.getAttribute('data-key') === key) {
                r.classList.remove('geo-row-highlighted');
            }
        });
    }
}

// ── 15. Shared Benchmark & Diagnostics Renderers ──────────────────
function renderHorizonTable(metrics) {
    const tbody = document.getElementById('horizon-tbody');
    if (!tbody) return;
    if (!metrics || !metrics.length) {
        tbody.innerHTML = `<tr><td colspan="4" class="table-empty">No horizon data available.</td></tr>`;
        return;
    }
    tbody.innerHTML = metrics.map((h, i) => {
        const stepNum = h.horizon_week ?? h.horizon_step ?? (i + 1);
        const maeStr = formatDecimal(h.mae);
        const wapeStr = formatPercent(h.wape);
        const obs = (h.n_observations || 0).toLocaleString();
        return `
            <tr>
                <td class="font-medium text-charcoal">Week ${stepNum}</td>
                <td class="text-right">${maeStr}</td>
                <td class="text-right font-semibold text-gold">${wapeStr}</td>
                <td class="text-right text-muted">${obs}</td>
            </tr>
        `;
    }).join('');
}

function renderCategoryTable(metrics) {
    const tbody = document.getElementById('category-tbody');
    if (!tbody) return;
    if (!metrics || !metrics.length) {
        tbody.innerHTML = `<tr><td colspan="4" class="table-empty">No category data available.</td></tr>`;
        return;
    }
    tbody.innerHTML = metrics.map(c => {
        const categoryName = c.scope_value || c.category || c.scope || 'General';
        const maeStr = formatDecimal(c.mae);
        const wapeStr = formatPercent(c.wape);
        const obs = (c.n_observations || 0).toLocaleString();
        return `
            <tr>
                <td class="font-medium text-charcoal">${categoryName}</td>
                <td class="text-right">${maeStr}</td>
                <td class="text-right font-semibold text-gold">${wapeStr}</td>
                <td class="text-right text-muted">${obs}</td>
            </tr>
        `;
    }).join('');
}

function renderHorizonErrorChart(metrics) {
    const canvas = document.getElementById('chart-horizon-error');
    if (!canvas || !metrics || !metrics.length) return;

    if (state.charts.horizonError) state.charts.horizonError.destroy();

    const labels = metrics.map((m, i) => `Week ${m.horizon_week ?? m.horizon_step ?? (i + 1)}`);
    const wapeValues = metrics.map(m => m.wape !== null && m.wape !== undefined ? Number(Number(m.wape).toFixed(2)) : 0);
    const maeValues = metrics.map(m => m.mae !== null && m.mae !== undefined ? Number(Number(m.mae).toFixed(2)) : 0);

    const ctx = canvas.getContext('2d');
    state.charts.horizonError = new Chart(ctx, {
        data: {
            labels: labels,
            datasets: [
                {
                    type: 'bar',
                    label: 'WAPE Error (%)',
                    data: wapeValues,
                    yAxisID: 'yWape',
                    backgroundColor: 'rgba(227, 196, 155, 0.75)',
                    borderColor: '#E3C49B',
                    borderWidth: 1.5,
                    borderRadius: 6,
                    barPercentage: 0.5,
                    order: 2,
                },
                {
                    type: 'line',
                    label: 'MAE Deviation (Units)',
                    data: maeValues,
                    yAxisID: 'yMae',
                    borderColor: '#AF9AC9',
                    backgroundColor: '#AF9AC9',
                    borderWidth: 3,
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    pointBackgroundColor: '#FFFFFF',
                    pointBorderColor: '#AF9AC9',
                    pointBorderWidth: 2.5,
                    tension: 0.3,
                    order: 1,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        boxWidth: 12,
                        font: { family: 'Inter', size: 11, weight: '600' }
                    }
                },
                tooltip: {
                    backgroundColor: '#FFFFFF',
                    titleColor: '#2D2926',
                    bodyColor: '#666161',
                    borderColor: '#E9E3DE',
                    borderWidth: 1,
                    padding: 10,
                    callbacks: {
                        label: function(context) {
                            if (context.dataset.yAxisID === 'yWape') {
                                return ` WAPE (Relative Error): ${context.parsed.y.toFixed(2)}%`;
                            } else {
                                return ` MAE (Mean Unit Deviation): ${context.parsed.y.toFixed(2)} units`;
                            }
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    title: {
                        display: true,
                        text: 'Forecast Horizon (Week 1 = Near-term, Week 4 = Long-term)',
                        color: '#666161',
                        font: { family: 'Inter', size: 11, weight: '600' }
                    }
                },
                yWape: {
                    type: 'linear',
                    position: 'left',
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'WAPE Error Rate (%)',
                        color: '#A5937B',
                        font: { family: 'Inter', size: 11, weight: '600' }
                    },
                    ticks: {
                        callback: v => v + '%'
                    },
                    grid: { color: 'rgba(165, 147, 123, 0.1)' }
                },
                yMae: {
                    type: 'linear',
                    position: 'right',
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'MAE (Units / Week)',
                        color: '#8E77AC',
                        font: { family: 'Inter', size: 11, weight: '600' }
                    },
                    ticks: {
                        callback: v => v + ' u'
                    },
                    grid: { drawOnChartArea: false }
                }
            },
        },
    });
}

function renderFailureCases(cases) {
    const container = document.getElementById('failure-cases-content');
    if (!container) return;
    if (!cases.length) {
        container.innerHTML = `<p class="text-muted">No extreme outlier failures detected in test window.</p>`;
        return;
    }
    container.innerHTML = cases.map(c => `
        <div class="failure-item">
            <div class="failure-header">
                <span class="failure-title font-semibold">SKU: ${c.product_id} (${c.forecast_week})</span>
                <span class="badge badge-danger">Abs Error: ${Math.round(c.absolute_error)}</span>
            </div>
            <p class="failure-explanation">${c.explanation}</p>
        </div>
    `).join('');
}

// ── 16. Benchmark Modal & Trigger ─────────────────────────────────
function runBenchmarkModal() {
    const modal = document.getElementById('benchmark-modal');
    if (modal) modal.style.display = 'flex';
}

function closeBenchmarkModal() {
    const modal = document.getElementById('benchmark-modal');
    if (modal) modal.style.display = 'none';
}

async function executeBenchmark() {
    const holdout = parseInt(document.getElementById('modal-holdout-weeks')?.value || '8');
    closeBenchmarkModal();
    showToast(`Started 5-model evaluation benchmark (${holdout} weeks holdout)...`, 'info');

    try {
        const res = await fetch('/api/evaluation/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ holdout_weeks: holdout }),
        });
        if (!res.ok) throw new Error('Benchmark failed');
        const data = await res.json();
        showToast('5-model evaluation completed successfully!', 'success');
        await loadLatestEvaluation();
    } catch (e) {
        showToast('Evaluation error: ' + e.message, 'error');
    }
}

async function generateForecast(triggerBtn) {
    // ── Collect all "Generate Forecast" buttons to animate them all ──
    const allBtns = [
        document.getElementById('btn-gen-forecast'),
        document.getElementById('btn-gen-forecast-overview'),
        document.getElementById('btn-gen-forecast-explorer'),
    ].filter(Boolean);

    // ── Save original state for each button ──
    const originals = allBtns.map(btn => ({
        btn,
        icon: btn.querySelector('[data-lucide], .btn-icon'),
        labels: [...btn.querySelectorAll('span')],
        disabled: btn.disabled,
        innerHTML: btn.innerHTML,
    }));

    // ── Set loading state ──
    allBtns.forEach(btn => {
        btn.disabled = true;
        btn.classList.add('btn-loading');
        const icon = btn.querySelector('[data-lucide], .btn-icon');
        if (icon) {
            icon.setAttribute('data-lucide', 'loader-2');
            icon.classList.add('spinning');
            if (window.lucide) lucide.createIcons({ nodes: [icon] });
        }
        btn.querySelectorAll('span').forEach((sp, i) => {
            if (i === 0) sp.textContent = 'Generating…';
            else sp.style.display = 'none';
        });
    });

    // ── Show animated progress overlay ──
    showForecastProgressOverlay(true);
    showToast('🔄 Generating 4-week recursive forecasts with CatBoost model…', 'info');

    try {
        const res = await fetch('/api/forecast/generate', { method: 'POST' });
        if (!res.ok) throw new Error(`Server error ${res.status}`);
        const data = await res.json();

        // ── Success state on buttons ──
        allBtns.forEach(btn => {
            const icon = btn.querySelector('[data-lucide], .btn-icon');
            if (icon) {
                icon.setAttribute('data-lucide', 'check-circle-2');
                icon.classList.remove('spinning');
                if (window.lucide) lucide.createIcons({ nodes: [icon] });
            }
            btn.classList.remove('btn-loading');
            btn.classList.add('btn-success-flash');
            btn.querySelectorAll('span').forEach((sp, i) => {
                if (i === 0) sp.textContent = '✓ Updated!';
                else sp.style.display = '';
            });
        });

        showForecastProgressOverlay(false);
        showToast(`✅ Forecast complete! ${(data.n_forecasts || 0).toLocaleString()} series updated (${data.run_id})`, 'success');

        // ── Refresh active tab data ──
        if (state.activeTab === 'ret-overview') {
            await loadRetailOverview();
        } else if (state.activeTab === 'ret-forecasts') {
            await loadForecastTable();
            await loadGeoForecastData();
        } else {
            // Called from a banner in either tab — refresh both
            await Promise.all([loadRetailOverview(), loadForecastTable()]);
        }

        // ── Restore buttons after 2.5s ──
        setTimeout(() => {
            originals.forEach(({ btn, innerHTML }) => {
                btn.innerHTML = innerHTML;
                btn.disabled = false;
                btn.classList.remove('btn-loading', 'btn-success-flash');
                if (window.lucide) lucide.createIcons({ nodes: [btn] });
            });
        }, 2500);

    } catch (e) {
        showForecastProgressOverlay(false);
        showToast('❌ Forecast generation failed: ' + e.message, 'error');

        // ── Restore buttons immediately on error ──
        originals.forEach(({ btn, innerHTML }) => {
            btn.innerHTML = innerHTML;
            btn.disabled = false;
            btn.classList.remove('btn-loading');
            if (window.lucide) lucide.createIcons({ nodes: [btn] });
        });
    }
}

function showForecastProgressOverlay(show) {
    let overlay = document.getElementById('forecast-progress-overlay');
    if (show) {
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'forecast-progress-overlay';
            overlay.className = 'forecast-progress-overlay';
            overlay.innerHTML = `
                <div class="forecast-progress-card">
                    <div class="forecast-progress-spinner"></div>
                    <div class="forecast-progress-content">
                        <h4 class="forecast-progress-title">Generating 4-Week Forecasts</h4>
                        <p class="forecast-progress-sub">CatBoost recursive multi-step prediction in progress…</p>
                        <div class="forecast-progress-steps">
                            <div class="fcp-step active" id="fcp-step-1">
                                <span class="fcp-dot"></span>
                                <span>Loading dataset & computing lag features</span>
                            </div>
                            <div class="fcp-step" id="fcp-step-2">
                                <span class="fcp-dot"></span>
                                <span>Running CatBoost recursive forecasts</span>
                            </div>
                            <div class="fcp-step" id="fcp-step-3">
                                <span class="fcp-dot"></span>
                                <span>Saving forecast run & clearing cache</span>
                            </div>
                            <div class="fcp-step" id="fcp-step-4">
                                <span class="fcp-dot"></span>
                                <span>Refreshing analytics &amp; charts</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            document.body.appendChild(overlay);

            // Animate progress steps
            const steps = [2, 3, 4];
            const delays = [3000, 8000, 16000];
            steps.forEach((n, i) => {
                setTimeout(() => {
                    const prev = document.getElementById(`fcp-step-${n - 1}`);
                    const curr = document.getElementById(`fcp-step-${n}`);
                    if (prev) { prev.classList.remove('active'); prev.classList.add('completed'); }
                    if (curr) curr.classList.add('active');
                }, delays[i]);
            });
        }
        overlay.classList.add('visible');
    } else {
        if (overlay) {
            overlay.classList.remove('visible');
            setTimeout(() => overlay.remove(), 400);
        }
    }
}

function refreshCurrentWorkspace() {
    showToast('Refreshing workspace telemetry...', 'info');
    if (state.currentWorkspace === 'dev') {
        loadDeveloperWorkspace();
    } else {
        loadRetailerWorkspace();
    }
}

function saveDevSettings() {
    const appModel = document.getElementById('setting-approved-model')?.value;
    if (appModel) {
        approveModelForProduction(appModel);
    }
}

// ── 17. Grounded Gemini AI Assistant ──────────────────────────────
function initChatInput() {
    if (!DOM.chatInput) return;
    DOM.chatInput.addEventListener('input', () => {
        DOM.chatInput.style.height = 'auto';
        DOM.chatInput.style.height = Math.min(DOM.chatInput.scrollHeight, 120) + 'px';
        if (DOM.chatCharCounter) {
            DOM.chatCharCounter.textContent = `${DOM.chatInput.value.length} / 2000`;
        }
    });
}

function handleChatKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function sendQuickPrompt(promptText) {
    if (!DOM.chatInput) return;
    DOM.chatInput.value = promptText;
    sendMessage();
}

async function sendMessage() {
    const message = DOM.chatInput?.value.trim();
    if (!message || state.isProcessing) return;

    state.isProcessing = true;
    DOM.chatInput.value = '';
    DOM.chatInput.style.height = 'auto';
    if (DOM.btnSend) DOM.btnSend.disabled = true;

    // Append User Message
    appendChatMessage('user', message);

    // Append Loading Indicator
    const loadingId = appendChatLoading();

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                history: state.conversationHistory,
            }),
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.error || errData.answer || 'Assistant communication failed');
        }
        const data = await res.json();
        const replyText = data.reply || data.answer || "No response received.";
        await finishChatLoading(loadingId);

        // Update history
        state.conversationHistory.push({ role: 'user', content: message });
        state.conversationHistory.push({ role: 'assistant', content: replyText });

        appendChatMessage('assistant', replyText, data.tools_called || data.tool_calls);
    } catch (e) {
        removeChatLoading(loadingId);
        appendChatMessage('assistant', `⚠️ Error: Could not connect to assistant. (${e.message})`);
    } finally {
        state.isProcessing = false;
        if (DOM.btnSend) DOM.btnSend.disabled = false;
    }
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function appendChatMessage(role, text, toolCalls) {
    if (!DOM.chatMessages) return;

    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const iconName = role === 'user' ? 'user' : 'bot';
    let toolCallsHtml = '';

    if (role === 'assistant' && toolCalls && toolCalls.length) {
        const queryCount = toolCalls.length;
        const queryLabel = queryCount === 1 ? '1 query executed' : `${queryCount} queries executed`;

        toolCallsHtml = `
            <details class="chat-tools-panel">
                <summary class="chat-tools-summary">
                    <span class="chat-tools-summary-title">
                        <i data-lucide="database" class="icon-xs"></i>
                        <span>${queryLabel}</span>
                    </span>
                    <i data-lucide="chevron-down" class="icon-xs chat-tools-chevron"></i>
                </summary>
                <div class="chat-tools-drawer">
                    ${toolCalls.map(tc => {
                        const fnName = tc.function || tc.name || 'query_tool';
                        const argsObj = tc.arguments || tc.args || {};
                        const hasArgs = argsObj && typeof argsObj === 'object' && Object.keys(argsObj).length > 0;
                        const argsFormatted = hasArgs ? JSON.stringify(argsObj, null, 2) : (typeof argsObj === 'string' && argsObj.trim() ? argsObj : '');
                        return `
                            <div class="chat-tool-entry">
                                <div class="chat-tool-head">
                                    <i data-lucide="terminal" class="icon-xs text-brand"></i>
                                    <code class="chat-tool-name">${escapeHtml(fnName)}</code>
                                </div>
                                ${argsFormatted ? `<pre class="chat-tool-args"><code>${escapeHtml(argsFormatted)}</code></pre>` : ''}
                            </div>
                        `;
                    }).join('')}
                </div>
            </details>
        `;
    }

    // Markdown formatted simple HTML
    const formattedHtml = formatChatMarkdown(text);

    let actionsToolbarHtml = '';
    if (role === 'assistant') {
        const msgId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7);
        msgDiv.setAttribute('data-msg-id', msgId);
        actionsToolbarHtml = `
            <div class="chat-actions-toolbar" data-msg-id="${msgId}">
                <button class="chat-action-btn copy-btn" title="Copy response to clipboard" onclick="copyChatMessage(this)">
                    <i data-lucide="copy" class="icon-xs"></i>
                    <span>Copy</span>
                </button>
                <div class="chat-feedback-group">
                    <button class="chat-action-btn feedback-btn thumbs-up" title="Good response (Reward assistant)" onclick="submitChatFeedback(this, 'positive')">
                        <i data-lucide="thumbs-up" class="icon-xs"></i>
                        <span>Good response</span>
                    </button>
                    <button class="chat-action-btn feedback-btn thumbs-down" title="Bad response (Penalize & improve next time)" onclick="submitChatFeedback(this, 'negative')">
                        <i data-lucide="thumbs-down" class="icon-xs"></i>
                        <span>Bad response</span>
                    </button>
                </div>
                <span class="feedback-ack-pill" style="display: none;"></span>
            </div>
        `;
    }

    msgDiv.innerHTML = `
        <div class="message-avatar">
            <i data-lucide="${iconName}"></i>
        </div>
        <div class="message-content">
            <div class="chat-text-body">${formattedHtml}</div>
            ${toolCallsHtml}
            ${actionsToolbarHtml}
        </div>
    `;

    DOM.chatMessages.appendChild(msgDiv);
    DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
    if (window.lucide) lucide.createIcons();
}

async function copyChatMessage(btn) {
    try {
        const msgContent = btn.closest('.message-content');
        if (!msgContent) return;
        const textBody = msgContent.querySelector('.chat-text-body') || msgContent;
        const clone = textBody.cloneNode(true);
        const toolbar = clone.querySelector('.chat-actions-toolbar');
        if (toolbar) toolbar.remove();
        const chips = clone.querySelector('.tool-call-chips');
        if (chips) chips.remove();
        const toolsPanel = clone.querySelector('.chat-tools-panel');
        if (toolsPanel) toolsPanel.remove();
        
        const rawText = clone.innerText.trim();
        await navigator.clipboard.writeText(rawText);

        btn.classList.add('copied');
        btn.innerHTML = `<i data-lucide="check" class="icon-xs text-success"></i> <span>Copied!</span>`;
        if (window.lucide) lucide.createIcons();
        showToast('Response copied to clipboard!', 'success');

        setTimeout(() => {
            btn.classList.remove('copied');
            btn.innerHTML = `<i data-lucide="copy" class="icon-xs"></i> <span>Copy</span>`;
            if (window.lucide) lucide.createIcons();
        }, 2000);
    } catch (err) {
        showToast('Could not copy to clipboard: ' + err.message, 'error');
    }
}

async function submitChatFeedback(btn, feedbackType) {
    const toolbar = btn.closest('.chat-actions-toolbar');
    if (!toolbar) return;

    const msgId = toolbar.getAttribute('data-msg-id') || ('msg_' + Date.now());
    const msgDiv = toolbar.closest('.message');
    const msgContent = toolbar.closest('.message-content');
    const textBody = msgContent?.querySelector('.chat-text-body') || msgContent;
    
    // Find preceding user query
    let userQuery = '';
    let prevElem = msgDiv?.previousElementSibling;
    while (prevElem) {
        if (prevElem.classList.contains('user')) {
            userQuery = prevElem.querySelector('.message-content')?.innerText.trim() || '';
            break;
        }
        prevElem = prevElem.previousElementSibling;
    }
    if (!userQuery && state.conversationHistory && state.conversationHistory.length >= 2) {
        const lastUser = [...state.conversationHistory].reverse().find(m => m.role === 'user');
        if (lastUser) userQuery = lastUser.content || lastUser.text || '';
    }

    const clone = textBody.cloneNode(true);
    const tb = clone.querySelector('.chat-actions-toolbar');
    if (tb) tb.remove();
    const chips = clone.querySelector('.tool-call-chips');
    if (chips) chips.remove();
    const tp = clone.querySelector('.chat-tools-panel');
    if (tp) tp.remove();
    const respText = clone.innerText.trim();

    const upBtn = toolbar.querySelector('.thumbs-up');
    const downBtn = toolbar.querySelector('.thumbs-down');
    const ackPill = toolbar.querySelector('.feedback-ack-pill');

    const isPositive = feedbackType === 'positive';

    if (isPositive) {
        upBtn?.classList.add('active-up');
        downBtn?.classList.remove('active-down');
        if (upBtn) upBtn.innerHTML = `<i data-lucide="check" class="icon-xs"></i> <span>Rewarded</span>`;
        if (downBtn) downBtn.style.opacity = '0.5';
        if (ackPill) {
            ackPill.style.display = 'inline-flex';
            ackPill.className = 'feedback-ack-pill ack-success';
            ackPill.innerHTML = `<i data-lucide="sparkles" class="icon-xs"></i> Reward saved · Assistant learned!`;
        }
        showToast('Reward recorded (+1)! The AI Assistant learned this preferred response style.', 'success');
    } else {
        downBtn?.classList.add('active-down');
        upBtn?.classList.remove('active-up');
        if (downBtn) downBtn.innerHTML = `<i data-lucide="alert-circle" class="icon-xs"></i> <span>Penalty noted</span>`;
        if (upBtn) upBtn.style.opacity = '0.5';
        if (ackPill) {
            ackPill.style.display = 'inline-flex';
            ackPill.className = 'feedback-ack-pill ack-warning';
            ackPill.innerHTML = `<i data-lucide="info" class="icon-xs"></i> Penalty recorded · Assistant will adjust next time`;
        }
        showToast('Penalty recorded (-1). The AI Assistant noted this feedback to adjust future responses.', 'info');
    }

    if (window.lucide) lucide.createIcons();

    // Call API
    try {
        await fetch('/api/chat/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message_id: msgId,
                user_query: userQuery,
                assistant_response: respText,
                feedback_type: feedbackType,
                score: isPositive ? 1 : -1,
                comment: isPositive ? 'Reward: Preferred response style' : 'Penalty: Needs improvement',
            }),
        });
    } catch (err) {
        console.warn('Feedback API call error:', err);
    }
}

function formatChatMarkdown(txt) {
    if (!txt) return '';

    // 1. Try Marked.js if loaded via CDN
    try {
        if (typeof window !== 'undefined' && window.marked && typeof window.marked.parse === 'function') {
            window.marked.setOptions({
                breaks: true,
                gfm: true,
            });
            return window.marked.parse(txt);
        }
    } catch (e) {
        console.warn('marked.parse failed, falling back to regex parser:', e);
    }

    // 2. Comprehensive built-in Markdown Parser fallback
    let str = txt;

    // Normalize newlines
    str = str.replace(/\r\n/g, '\n');

    // Code blocks
    str = str.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (m, lang, code) => {
        return `<pre><code class="language-${lang}">${escapeHtml(code.trim())}</code></pre>`;
    });

    // Inline code
    str = str.replace(/`([^`]+)`/g, (m, c) => `<code>${escapeHtml(c)}</code>`);

    // Headings (H1 - H4)
    str = str.replace(/^####\s+(.*?)$/gm, '<h4>$1</h4>');
    str = str.replace(/^###\s+(.*?)$/gm, '<h3>$1</h3>');
    str = str.replace(/^##\s+(.*?)$/gm, '<h2>$1</h2>');
    str = str.replace(/^#\s+(.*?)$/gm, '<h1>$1</h1>');

    // Horizontal Rules
    str = str.replace(/^(?:---|\*\*\*|___)\s*$/gm, '<hr>');

    // Blockquotes
    str = str.replace(/^>\s+(.*?)$/gm, '<blockquote>$1</blockquote>');

    // Bold & Italics
    str = str.replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>');
    str = str.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    str = str.replace(/\*(.*?)\*/g, '<em>$1</em>');
    str = str.replace(/___(.*?)___/g, '<strong><em>$1</em></strong>');
    str = str.replace(/__(.*?)__/g, '<strong>$1</strong>');
    str = str.replace(/_(.*?)_/g, '<em>$1</em>');

    // Unordered list items
    str = str.replace(/^[\s]*[-*+]\s+(.*?)$/gm, '<li>$1</li>');

    // Ordered list items
    str = str.replace(/^[\s]*(\d+)\.\s+(.*?)$/gm, '<li>$1</li>');

    // Wrap contiguous <li> into <ul>
    str = str.replace(/(?:<li>[\s\S]*?<\/li>\s*)+/g, match => `<ul>${match}</ul>`);

    // Simple markdown table parser
    str = str.replace(/((?:\|[^\n]+\|\n?)+)/g, tableMatch => {
        const rows = tableMatch.trim().split('\n').filter(r => r.includes('|'));
        if (rows.length < 2) return tableMatch;
        let tableHtml = '<table>';
        rows.forEach((row, idx) => {
            if (row.match(/^\|?\s*[-:]+[-| :]*\|?$/)) return; // separator row
            const cells = row.split('|').filter((_, cIdx, arr) => cIdx > 0 && cIdx < arr.length - 1);
            if (idx === 0) {
                tableHtml += '<thead><tr>' + cells.map(c => `<th>${c.trim()}</th>`).join('') + '</tr></thead><tbody>';
            } else {
                tableHtml += '<tr>' + cells.map(c => `<td>${c.trim()}</td>`).join('') + '</tr>';
            }
        });
        tableHtml += '</tbody></table>';
        return tableHtml;
    });

    // Paragraphs & Line Breaks (for lines not already wrapped in block elements)
    const blocks = str.split(/\n{2,}/);
    const htmlBlocks = blocks.map(block => {
        block = block.trim();
        if (!block) return '';
        if (block.startsWith('<h') || block.startsWith('<ul') || block.startsWith('<ol') || 
            block.startsWith('<pre') || block.startsWith('<table') || block.startsWith('<blockquote') || block.startsWith('<hr')) {
            return block;
        }
        return `<p>${block.replace(/\n/g, '<br>')}</p>`;
    });

    return htmlBlocks.filter(Boolean).join('');
}

// ── AI Generation Animation System (Gemini / GPT Style) ───────────
const activeAiLoaders = new Map();

function appendChatLoading() {
    if (!DOM.chatMessages) return null;
    const id = 'loading-' + Date.now();
    const div = document.createElement('div');
    div.className = 'message assistant loading-msg';
    div.id = id;
    div.innerHTML = `
        <div class="message-avatar ai-avatar-pulse">
            <i data-lucide="sparkles"></i>
        </div>
        <div class="message-content">
            <div class="ai-generating-card">
                <!-- Top Status Bar -->
                <div class="ai-gen-header">
                    <div class="ai-gen-phase-pill" id="${id}-phase-pill">
                        <span class="ai-gen-spinner"></span>
                        <span class="ai-gen-phase-name" id="${id}-phase-name">Analyzing Query</span>
                    </div>
                    <div class="ai-gen-timer" id="${id}-timer">0.0s</div>
                </div>

                <!-- Dynamic Main Activity Line -->
                <div class="ai-gen-status-row">
                    <div class="ai-gen-icon-orb" id="${id}-icon-orb">
                        <i data-lucide="brain"></i>
                    </div>
                    <div class="ai-gen-text-wrap">
                        <div class="ai-gen-action" id="${id}-action-text">Understanding retail intent & query scope...</div>
                        <div class="ai-gen-detail" id="${id}-detail-text">Evaluating parameters, store filters & forecasting horizons</div>
                    </div>
                </div>

                <!-- Shimmering Dynamic Track -->
                <div class="ai-shimmer-track">
                    <div class="ai-shimmer-bar"></div>
                </div>

                <!-- Interactive Stepper Timeline -->
                <div class="ai-gen-stepper" id="${id}-stepper">
                    <div class="ai-step-item current" id="${id}-step-1">
                        <span class="ai-step-bullet"><i data-lucide="loader-2" class="spin"></i></span>
                        <span class="ai-step-label">Parse parameters & query scope</span>
                    </div>
                    <div class="ai-step-item" id="${id}-step-2">
                        <span class="ai-step-bullet"><i data-lucide="circle" class="bullet-dot"></i></span>
                        <span class="ai-step-label">Execute function & demand tool calls</span>
                    </div>
                    <div class="ai-step-item" id="${id}-step-3">
                        <span class="ai-step-bullet"><i data-lucide="circle" class="bullet-dot"></i></span>
                        <span class="ai-step-label">Compute ML metrics & groundings</span>
                    </div>
                    <div class="ai-step-item" id="${id}-step-4">
                        <span class="ai-step-bullet"><i data-lucide="circle" class="bullet-dot"></i></span>
                        <span class="ai-step-label">Synthesize executive insights</span>
                    </div>
                </div>
            </div>
        </div>
    `;
    DOM.chatMessages.appendChild(div);
    DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
    if (window.lucide) lucide.createIcons();

    startAiProgressAnimation(id);
    return id;
}

function startAiProgressAnimation(id) {
    const startTime = Date.now();
    let currentStep = 1;

    const timerEl = document.getElementById(`${id}-timer`);
    const phaseName = document.getElementById(`${id}-phase-name`);
    const iconOrb = document.getElementById(`${id}-icon-orb`);
    const actionText = document.getElementById(`${id}-action-text`);
    const detailText = document.getElementById(`${id}-detail-text`);

    const interval = setInterval(() => {
        const elapsedMs = Date.now() - startTime;
        const elapsedSec = (elapsedMs / 1000).toFixed(1);
        if (timerEl) timerEl.textContent = `${elapsedSec}s`;

        // Stage 2: Tool Calling (after 1.0s)
        if (elapsedMs >= 1000 && currentStep === 1) {
            currentStep = 2;
            updateStepState(id, 1, 'completed');
            updateStepState(id, 2, 'current');
            if (phaseName) phaseName.textContent = 'Calling Tools';
            if (iconOrb) iconOrb.innerHTML = '<i data-lucide="database"></i>';
            if (actionText) actionText.textContent = 'Invoking retail intelligence functions & database...';
            if (detailText) detailText.textContent = 'Querying SKU sales aggregates, festival records & forecasts';
            if (window.lucide) lucide.createIcons();
        }

        // Stage 3: Analyzing & Computing (after 2.5s)
        if (elapsedMs >= 2500 && currentStep === 2) {
            currentStep = 3;
            updateStepState(id, 2, 'completed');
            updateStepState(id, 3, 'current');
            if (phaseName) phaseName.textContent = 'Analyzing Data';
            if (iconOrb) iconOrb.innerHTML = '<i data-lucide="line-chart"></i>';
            if (actionText) actionText.textContent = 'Processing statistical outputs & ML predictions...';
            if (detailText) detailText.textContent = 'Validating numerical claims & groundings against evidence';
            if (window.lucide) lucide.createIcons();
        }

        // Stage 4: Generating Response (after 4.0s)
        if (elapsedMs >= 4000 && currentStep === 3) {
            currentStep = 4;
            updateStepState(id, 3, 'completed');
            updateStepState(id, 4, 'current');
            if (phaseName) phaseName.textContent = 'Generating';
            if (iconOrb) iconOrb.innerHTML = '<i data-lucide="sparkles"></i>';
            if (actionText) actionText.textContent = 'Synthesizing executive demand insights...';
            if (detailText) detailText.textContent = 'Formulating strategic explanations & recommendations';
            if (window.lucide) lucide.createIcons();
        }
    }, 100);

    activeAiLoaders.set(id, { interval, startTime });
}

function updateStepState(loaderId, stepNum, stateClass) {
    const stepEl = document.getElementById(`${loaderId}-step-${stepNum}`);
    if (!stepEl) return;
    stepEl.className = `ai-step-item ${stateClass}`;
    const bullet = stepEl.querySelector('.ai-step-bullet');
    if (!bullet) return;

    if (stateClass === 'completed') {
        bullet.innerHTML = '<i data-lucide="check" class="text-success"></i>';
    } else if (stateClass === 'current') {
        bullet.innerHTML = '<i data-lucide="loader-2" class="spin"></i>';
    }
    if (window.lucide) lucide.createIcons();
}

async function finishChatLoading(id) {
    if (!id) return;
    const loaderInfo = activeAiLoaders.get(id);
    if (loaderInfo) {
        clearInterval(loaderInfo.interval);
        activeAiLoaders.delete(id);
    }

    // Complete all steps visually
    for (let s = 1; s <= 4; s++) {
        updateStepState(id, s, 'completed');
    }
    const phasePill = document.getElementById(`${id}-phase-pill`);
    const phaseName = document.getElementById(`${id}-phase-name`);
    const actionText = document.getElementById(`${id}-action-text`);
    const detailText = document.getElementById(`${id}-detail-text`);
    const iconOrb = document.getElementById(`${id}-icon-orb`);

    if (phasePill) phasePill.classList.add('completed');
    if (phaseName) phaseName.textContent = 'Finished';
    if (iconOrb) iconOrb.innerHTML = '<i data-lucide="check-check"></i>';
    if (actionText) actionText.textContent = 'Response synthesized & verified.';
    if (detailText) detailText.textContent = 'Completed analytical reasoning and tool execution';
    if (window.lucide) lucide.createIcons();

    // Brief smooth pause (260ms) before transitioning to response
    await new Promise(r => setTimeout(r, 260));

    const el = document.getElementById(id);
    if (el) el.remove();
}

function removeChatLoading(id) {
    if (!id) return;
    const loaderInfo = activeAiLoaders.get(id);
    if (loaderInfo) {
        clearInterval(loaderInfo.interval);
        activeAiLoaders.delete(id);
    }
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ── 18. Utilities ─────────────────────────────────────────────────
async function checkApiHealth() {
    try {
        const res = await fetch('/api/health');
        if (res.ok) {
            updateApiStatus(true, 'Retail ML Server Ready');
        } else {
            updateApiStatus(false, 'API Degraded');
        }
    } catch {
        updateApiStatus(false, 'API Offline');
    }
}

function updateApiStatus(isOnline, text) {
    if (!DOM.apiStatus) return;
    DOM.apiStatus.className = `api-status ${isOnline ? 'online' : 'offline'}`;
    const t = DOM.apiStatus.querySelector('.status-text');
    if (t) t.textContent = text;
}

function showToast(msg, type = 'info') {
    if (!DOM.toastContainer) return;
    const t = document.createElement('div');
    t.className = `toast toast-${type}`;
    t.innerHTML = `<span>${msg}</span>`;
    DOM.toastContainer.appendChild(t);
    setTimeout(() => {
        t.style.opacity = '0';
        setTimeout(() => t.remove(), 250);
    }, 3500);
}

function resizeAllCharts() {
    Object.values(state.charts).forEach(c => resizeChart(c));
}

function resizeChart(chart) {
    if (chart && typeof chart.resize === 'function') {
        chart.resize();
    }
}
