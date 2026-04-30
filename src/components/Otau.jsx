import React from 'react';

export const Otau = ({ index, count, isEnabled, isTuzdyk, owner, onClick, label, isHighlighted, animPhase }) => {
    let className = 'otau';
    if (isEnabled && count > 0) className += ' enabled';
    else className += ' disabled';

    if (isTuzdyk) {
        className += owner === 0 ? ' tuzdyk-p1' : ' tuzdyk-p2';
    }

    if (isHighlighted) {
        if (animPhase === 'pickup') className += ' anim-pickup';
        else if (animPhase === 'sow') className += ' anim-sow';
        else if (animPhase === 'capture') className += ' anim-capture';
        else if (animPhase === 'tuzdyk') className += ' anim-tuzdyk';
    }

    const stones = [];
    const maxVisibleStones = Math.min(count, 14);
    for (let i = 0; i < maxVisibleStones; i++) {
        stones.push(<span key={i} className="kumalak" />);
    }

    return (
        <div className="otau-wrapper">
            <span className="pocket-label top-label">{index >= 9 ? label : ''}</span>

            <button
                className={className}
                onClick={() => (isEnabled && count > 0) ? onClick(index) : null}
                disabled={!isEnabled || count === 0}
                aria-label={`Отау ${label}, ${count} камней`}
            >
                {isTuzdyk ? (
                    <span className="tuzdyk-flag">
                        <span className="tuzdyk-icon">Т</span>
                    </span>
                ) : (
                    <span className="kumalak-grid">
                        {stones}
                        {count > 14 && <span className="plus-more">+{count - 14}</span>}
                    </span>
                )}
                <span className="count-badge">{isTuzdyk ? 'Т' : count}</span>
            </button>

            <span className="pocket-label bottom-label">{index < 9 ? label : ''}</span>
        </div>
    );
};
