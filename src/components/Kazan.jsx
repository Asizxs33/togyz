import React from 'react';

export const Kazan = ({ player, count, isActive }) => {
    const className = `kazan kazan-p${player + 1} ${isActive ? `active-p${player + 1}` : ''}`;

    const stones = [];
    const maxVisibleStones = Math.min(count, 38);
    for (let i = 0; i < maxVisibleStones; i++) {
        stones.push(<span key={i} className="kumalak kazan-kumalak" />);
    }

    return (
        <div className="kazan-wrapper">
            <div className="kazan-score-box">
                <span>{player === 0 ? 'Вы' : 'AI'}</span>
                <strong>{count}</strong>
            </div>

            <div className={className}>
                <div className="kumalak-line">
                    {stones}
                    {count > 38 && <span className="plus-more kazan-plus">+{count - 38}</span>}
                </div>
            </div>
        </div>
    );
};
